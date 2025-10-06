# src/trainer.py
from tqdm import tqdm
import torch
import numpy as np
from sklearn.metrics import accuracy_score, jaccard_score
import time
import random
from thop import profile

class Trainer:
    def __init__(self, model, optimizer, scheduler, criterion, train_loader, val_loader, manager, device, config):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.manager = manager
        self.device = device
        self.config = config
        self.scaler = torch.amp.GradScaler(enabled=(self.device.type == 'cuda'))
        self.start_epoch = 0
        self.best_miou = 0.0
        self.gflops = 0.0
        self.params_m = 0.0

        # Calculate and store model stats once upon initialization
        self._calculate_and_store_model_stats()
        
    def _calculate_and_store_model_stats(self):
        """Calculates and prints GFLOPs and total parameters, storing them in the instance."""
        print("\n--- Calculating Model Statistics ---")
        self.model.eval()
        try:
            patch_size = self.config.get('data', {}).get('patching', {}).get('patch_size', 224)
            model_params = self.config.get('model', {}).get('params', {})
            
            # Determine in_channels safely
            if 'num_selected_bands' in model_params and model_params['num_selected_bands'] is not None:
                in_channels = model_params['num_selected_bands']
            elif 'in_channels' in model_params:
                in_channels = model_params['in_channels']
            else:
                print("Warning: Could not determine 'in_channels' for stats calculation. Falling back to 3.")
                in_channels = 3

            dummy_input = torch.randn(1, in_channels, patch_size, patch_size).to(self.device)
            
            macs, params = profile(self.model, inputs=(dummy_input,), verbose=False)
            
            self.gflops = (macs * 2) / 1e9  # Convert MACs to GFLOPs
            self.params_m = params / 1e6  # Convert to Millions
            
            print(f"✓ Model Stats: {self.gflops:.2f} GFLOPs, {self.params_m:.2f} M Params")

        except Exception as e:
            print(f"  - Error during model stats calculation: {e}. Stats will be 0.")
            self.gflops = 0.0
            self.params_m = 0.0
        self.model.train() # Return model to training mode

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
        
        for epoch in range(self.start_epoch, self.config['training']['num_epochs']):
            epoch_num = epoch + 1
            print(f"\n--- Epoch {epoch_num}/{self.config['training']['num_epochs']} ---")
            
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
            
            checkpoint_state = {
                'epoch': epoch_num, 
                'model_state_dict': self.model.state_dict(), 
                'optimizer_state_dict': self.optimizer.state_dict(), 
                'scheduler_state_dict': self.scheduler.state_dict(), 
                'best_miou': self.best_miou,
                'scaler_state_dict': self.scaler.state_dict()
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
                loss = self.criterion(outputs, masks)
            
            self.scaler.scale(loss).backward()
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
                    loss = self.criterion(outputs, masks)
                
                total_loss += loss.item()
                
                # Handle dict or tensor output for evaluation
                if isinstance(outputs, dict):
                    seg_logits = outputs['segmentation']
                else:
                    seg_logits = outputs

                preds = torch.argmax(seg_logits, dim=1)
                all_preds.append(preds.cpu().numpy())
                all_trues.append(masks.cpu().numpy())

                if len(saved_samples) < num_samples_to_save:
                    batch_size = images.shape[0]
                    for i in range(batch_size):
                        if len(saved_samples) < num_samples_to_save:
                            img_np = images[i].permute(1, 2, 0).cpu().numpy()
                            mask_np = masks[i].cpu().numpy()
                            pred_np = preds[i].cpu().numpy()
                            saved_samples.append((img_np, mask_np, pred_np))
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
        
        try:
            dummy_input, _ = next(iter(self.val_loader))
            dummy_input = dummy_input.to(self.device)
        except StopIteration:
            print("Warning: Validation loader is empty. Cannot benchmark. Using random data.")
            bs = self.config['training']['batch_size']
            model_params = self.config.get('model', {}).get('params', {})
            if 'num_selected_bands' in model_params and model_params['num_selected_bands'] is not None:
                c = model_params['num_selected_bands']
            else:
                c = model_params.get('in_channels', 3)
            h = w = self.config.get('data', {}).get('patching', {}).get('patch_size', 224)
            dummy_input = torch.randn(bs, c, h, w, device=self.device)

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

    def _run_final_analysis(self):
        print("\n4. Final Evaluation and Analysis...")
        print("   Loading best model for detailed report...")
        
        checkpoint = self.manager.load_checkpoint(filename="best_model.pth")
        if checkpoint:
            self.model.load_state_dict(torch.load(self.manager.models_dir / "best_model.pth", map_location=self.device))
        else:
            print("Warning: best_model.pth not found. Using the last model state.")

        # Get performance metrics (FPS, Latency)
        performance_metrics = self.benchmark_performance()
        
        # Add pre-calculated stats to the metrics dictionary
        performance_metrics['gflops'] = self.gflops
        performance_metrics['params_m'] = self.params_m

        all_preds, all_trues = [], []
        with torch.no_grad():
            for images, masks in tqdm(self.val_loader, desc="Final Evaluation"):
                outputs = self.model(images.to(self.device))
                
                # Handle dict or tensor output for final analysis
                if isinstance(outputs, dict):
                    main_output = outputs['segmentation']
                else:
                    main_output = outputs

                all_preds.append(torch.argmax(main_output, dim=1).cpu().numpy())
                all_trues.append(masks.numpy())
                
        flat_preds = np.concatenate([p.flatten() for p in all_preds])
        flat_trues = np.concatenate([t.flatten() for t in all_trues])
        
        # Calculate efficiency
        efficiency = self.best_miou / self.gflops if self.gflops > 0 else 0
        performance_metrics['efficiency'] = efficiency

        self.manager.generate_final_report(self.best_miou, flat_preds, flat_trues, self.config, performance_metrics)
        
        # Write a simple final_summary.txt for quick checks
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