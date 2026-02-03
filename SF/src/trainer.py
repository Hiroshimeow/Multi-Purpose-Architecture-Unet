# src/trainer.py
from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, jaccard_score
import time
import random
from thop import profile

class Trainer:
    def __init__(self, model, optimizer, scheduler, criterion, train_loader, val_loader, manager, device, config, num_epochs: int):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.manager = manager
        self.device = device
        self.config = config
        self.num_epochs = num_epochs
        
        # Safe CUDA/cuDNN initialization
        if torch.cuda.is_available():
            torch.backends.cudnn.enabled = False
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            print("   Disabled cuDNN to avoid CUDNN_STATUS_NOT_INITIALIZED error.")

        self.scaler = torch.amp.GradScaler(enabled=(self.device.type == 'cuda'))
        self.start_epoch = 0
        self.best_miou = 0.0
        self.gflops = 0.0
        self.params_m = 0.0

        self.use_sr_head = self.config.get('model', {}).get('params', {}).get('use_sr_head', False)
        self.recon_lambda = self.config.get('training', {}).get('recon_lambda', 0.1)

        self._calculate_and_store_model_stats()
        
    def _calculate_and_store_model_stats(self):
        """Calculates and prints GFLOPs and total parameters, storing them in the instance."""
        print("\n--- Calculating Model Statistics ---")
        self.model.eval()
        
        original_sr_state = False
        if self.use_sr_head:
            if hasattr(self.model, 'segmentation_network') and hasattr(self.model.segmentation_network, 'use_sr_head'):
                original_sr_state = self.model.segmentation_network.use_sr_head
                self.model.segmentation_network.use_sr_head = False
            elif hasattr(self.model, 'use_sr_head'):
                original_sr_state = self.model.use_sr_head
                self.model.use_sr_head = False

        try:
            patch_size = self.config.get('data', {}).get('patching', {}).get('patch_size', 224)
            model_cfg = self.config.get('model', {})
            model_params = model_cfg.get('params', {})
            model_name = model_cfg.get('name', '')

            # FIX: Determine in_channels for dummy_input correctly
            if model_name == 'TABS':
                # ASRAN_LBS always takes the full channel count as input for the band selector module
                in_channels = model_params['in_channels']
            else:
                # For baseline models, 'in_channels' is explicitly set to 'k'. Prioritize it.
                if 'in_channels' in model_params:
                    in_channels = model_params['in_channels']
                elif 'num_select_bands' in model_params and model_params['num_select_bands'] is not None:
                    in_channels = model_params['num_select_bands']
                else:
                    in_channels = 3 # Fallback

            dummy_input = torch.randn(1, in_channels, patch_size, patch_size).to(self.device)
            macs, params = profile(self.model, inputs=(dummy_input,), verbose=False)
            self.gflops = (macs * 2) / 1e9
            self.params_m = params / 1e6
            print(f"✓ Model Stats: {self.gflops:.2f} GFLOPs, {self.params_m:.2f} M Params")

        except Exception as e:
            print(f"  - Error during model stats calculation: {e}. Stats will be 0.")
            self.gflops = 0.0
            self.params_m = 0.0
        finally:
            if self.use_sr_head:
                if hasattr(self.model, 'segmentation_network') and hasattr(self.model.segmentation_network, 'use_sr_head'):
                    self.model.segmentation_network.use_sr_head = original_sr_state
                elif hasattr(self.model, 'use_sr_head'):
                    self.model.use_sr_head = original_sr_state
            self.model.train()

    def _load_state(self):
        if self.manager.checkpoint_exists("latest_checkpoint.pth"):
            state = self.manager.load_checkpoint("latest_checkpoint.pth")
            if state:
                self.model.load_state_dict(state['model_state_dict'])
                self.optimizer.load_state_dict(state['optimizer_state_dict'])
                self.scheduler.load_state_dict(state['scheduler_state_dict'])
                self.start_epoch = state['epoch']
                self.best_miou = state['best_miou']
                if 'scaler_state_dict' in state:
                    self.scaler.load_state_dict(state['scaler_state_dict'])
                print(f"✓ Resumed training from epoch {self.start_epoch}.")

    def train(self):
        self._load_state()
        patience_counter = 0
        for epoch in range(self.start_epoch, self.num_epochs):
            epoch_num = epoch + 1
            
            # --- Temperature Annealing for LearnableBandSelector (TABS) ---
            if hasattr(self.model, 'band_selector') and self.model.band_selector is not None:
                # Anneal from 5.0 to 0.1 over 100 epochs
                # Formula: tau = initial * exp(-gamma * epoch)
                # 0.1 = 5.0 * exp(-gamma * 100) => gamma = 0.03912
                if epoch < 100:
                    gamma = 0.03912
                    new_tau = 5.0 * np.exp(-gamma * epoch)
                else:
                    new_tau = 0.1
                
                # Update model's band selector temperature
                self.model.band_selector.temperature = new_tau
                # Optional: Log it
                # print(f"  > Epoch {epoch_num}: Updated Gumbel Temperature to {new_tau:.4f}")

            print(f"\n--- Epoch {epoch_num}/{self.num_epochs} ---")
            train_loss = self._train_one_epoch()
            val_loss, val_acc, val_miou, val_samples = self._evaluate()
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()
            metrics = {'epoch': epoch_num, 'train_loss': train_loss, 'val_loss': val_loss, 'val_acc': val_acc, 'val_miou': val_miou, 'lr': self.optimizer.param_groups[0]['lr']}
            self.manager.log_epoch(metrics)
            print(f"Epoch {epoch_num} Summary: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val mIoU: {val_miou:.4f}")
            is_best = val_miou > self.best_miou
            if is_best:
                self.best_miou = val_miou
                patience_counter = 0
                print(f"✨ New best mIoU: {val_miou:.4f}. Saving model and prediction samples...")
                if val_samples:
                    self.manager.save_prediction_samples(val_samples)
            else:
                patience_counter += 1
            
            # Prepare rich metadata checkpoint
            checkpoint_state = {
                'epoch': epoch_num,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'best_miou': self.best_miou,
                'scaler_state_dict': self.scaler.state_dict(),
                'config': self.config,
                'seed': self.config.get('data', {}).get('seed', None)
            }
            self.manager.save_checkpoint(state=checkpoint_state, is_best=is_best)
            
            if patience_counter >= self.config['training']['early_stopping_patience']:
                print(f"🛑 Early stopping triggered after {patience_counter} epochs without improvement.")
                break
        self._run_final_analysis()

    def _train_one_epoch(self):
        self.model.train()
        total_loss = 0.0
        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        for images, masks in pbar:
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=self.device.type, dtype=torch.float16, enabled=(self.device.type == 'cuda')):
                outputs = self.model(images)
                
                # Generic handling of dictionary outputs
                if isinstance(outputs, dict):
                    seg_logits = outputs.get('segmentation', outputs) # Fallback to outputs if key missing (unlikely)
                    if 'reconstruction' in outputs and self.use_sr_head:
                         seg_loss = self.criterion(seg_logits, masks)
                         recon_loss = F.mse_loss(outputs['reconstruction'], outputs['original_input'])
                         loss = seg_loss + self.recon_lambda * recon_loss
                    else:
                         loss = self.criterion(seg_logits, masks)
                else:
                    seg_logits = outputs
                    loss = self.criterion(outputs, masks)

            self.scaler.scale(loss).backward()
            
            # --- Added Gradient Clipping ---
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        return total_loss / len(self.train_loader)
    
    def _evaluate(self, num_samples_to_save=8):
        self.model.eval()
        total_loss = 0.0
        all_preds, all_trues = [], []
        saved_samples = []
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validating", leave=False)
            for images, masks in pbar:
                images = images.to(self.device, non_blocking=True)
                masks = masks.to(self.device, non_blocking=True)
                with torch.amp.autocast(device_type=self.device.type, dtype=torch.float16, enabled=(self.device.type == 'cuda')):
                    outputs = self.model(images)
                    
                    # Generic handling of dictionary outputs
                    if isinstance(outputs, dict):
                        seg_logits = outputs.get('segmentation', outputs)
                        if 'reconstruction' in outputs and self.use_sr_head:
                             seg_loss = self.criterion(seg_logits, masks)
                             recon_loss = F.mse_loss(outputs['reconstruction'], outputs['original_input'])
                             loss = seg_loss + self.recon_lambda * recon_loss
                        else:
                             loss = self.criterion(seg_logits, masks)
                    else:
                        seg_logits = outputs
                        loss = self.criterion(outputs, masks)

                total_loss += loss.item()
                preds = torch.argmax(seg_logits, dim=1)
                all_preds.append(preds.cpu().numpy())
                all_trues.append(masks.cpu().numpy())
                if len(saved_samples) < num_samples_to_save:
                    # Move tensors to CPU before saving
                    images_cpu = images.cpu()
                    masks_cpu = masks.cpu()
                    preds_cpu = preds.cpu()
                    for i in range(images_cpu.shape[0]):
                        if len(saved_samples) < num_samples_to_save:
                            saved_samples.append((images_cpu[i], masks_cpu[i], preds_cpu[i]))
                        else:
                            break
        random.shuffle(saved_samples)
        flat_preds = np.concatenate([p.flatten() for p in all_preds])
        flat_trues = np.concatenate([t.flatten() for t in all_trues])
        accuracy = accuracy_score(flat_trues, flat_preds)
        miou = jaccard_score(flat_trues, flat_preds, average='macro', zero_division=0)
        return total_loss / len(self.val_loader), accuracy, miou, saved_samples

    def benchmark_performance(self, num_warmup=50, num_runs=200):
        print("\n--- Benchmarking Performance ---")
        self.model.eval()
        original_sr_state = False
        if self.use_sr_head:
            if hasattr(self.model, 'segmentation_network') and hasattr(self.model.segmentation_network, 'use_sr_head'):
                original_sr_state = self.model.segmentation_network.use_sr_head
                self.model.segmentation_network.use_sr_head = False
            elif hasattr(self.model, 'use_sr_head'):
                original_sr_state = self.model.use_sr_head
                self.model.use_sr_head = False
        try:
            dummy_input, _ = next(iter(self.val_loader))
            dummy_input = dummy_input.to(self.device)
            print(f"Input tensor shape for benchmark: {dummy_input.shape}")
            print(f"Running {num_warmup} warmup iterations...")
            with torch.no_grad():
                for _ in range(num_warmup):
                    _ = self.model(dummy_input)
            if self.device.type == 'cuda': torch.cuda.synchronize()
            print(f"Running {num_runs} benchmark iterations...")
            start_time = time.time()
            with torch.no_grad():
                for _ in range(num_runs):
                    _ = self.model(dummy_input)
            if self.device.type == 'cuda': torch.cuda.synchronize()
            end_time = time.time()
            total_time = end_time - start_time
            total_images = num_runs * dummy_input.shape[0]
            fps = total_images / total_time
            avg_latency_ms = (total_time / num_runs) * 1000
            print(f"✓ Benchmark complete.")
            print(f"  - Average Latency per Batch: {avg_latency_ms:.2f} ms")
            print(f"  - Frames Per Second (FPS): {fps:.2f}")
            return {'fps': fps, 'latency_ms': avg_latency_ms}
        except Exception as e:
            print(f"Error during benchmark: {e}")
            return {'fps': 0, 'latency_ms': 0}
        finally:
            if self.use_sr_head:
                if hasattr(self.model, 'segmentation_network') and hasattr(self.model.segmentation_network, 'use_sr_head'):
                    self.model.segmentation_network.use_sr_head = original_sr_state
                elif hasattr(self.model, 'use_sr_head'):
                    self.model.use_sr_head = original_sr_state

    def _run_final_analysis(self):
        print("\n4. Final Evaluation and Analysis...")
        print("   Loading best model for detailed report...")
        checkpoint = self.manager.load_checkpoint(filename="best_model.pth")
        if checkpoint:
            self.model.load_state_dict(torch.load(self.manager.models_dir / "best_model.pth", map_location=self.device))
        else:
            print("Warning: best_model.pth not found. Using the last model state.")
        performance_metrics = self.benchmark_performance()
        performance_metrics['gflops'] = self.gflops
        performance_metrics['params_m'] = self.params_m
        all_preds, all_trues = [], []
        with torch.no_grad():
            for images, masks in tqdm(self.val_loader, desc="Final Evaluation"):
                outputs = self.model(images.to(self.device))
                if isinstance(outputs, dict):
                    main_output = outputs['segmentation']
                else:
                    main_output = outputs
                all_preds.append(torch.argmax(main_output, dim=1).cpu().numpy())
                all_trues.append(masks.numpy())
        flat_preds = np.concatenate([p.flatten() for p in all_preds])
        flat_trues = np.concatenate([t.flatten() for t in all_trues])
        efficiency = self.best_miou / self.gflops if self.gflops > 0 else 0
        performance_metrics['efficiency'] = efficiency
        self.manager.generate_final_report(self.best_miou, flat_preds, flat_trues, self.config, performance_metrics)
        try:
            summary_path = self.manager.output_dir / "final_summary.txt"
            with open(summary_path, 'w') as f:
                f.write(f"Best Validation mIoU: {self.best_miou:.4f}\n")
                f.write(f"FPS: {performance_metrics.get('fps', 0):.2f}\n")
                f.write(f"Latency (ms): {performance_metrics.get('latency_ms', 0):.2f}\n")
                f.write(f"GFLOPs: {self.gflops:.2f}\n")
                f.write(f"Parameters (M): {self.params_m:.2f}\n")
        except Exception as e:
            print(f"Warning: Could not write final_summary.txt. Error: {e}")
        print(f"\n✅ Analysis complete. All results saved to '{self.manager.output_dir}' directory.")