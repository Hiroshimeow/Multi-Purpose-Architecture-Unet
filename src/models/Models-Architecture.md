### **1. Unet-Base**

*   **Parameters**: `in_channels=25`, `initial_filters=32`, `num_classes=5`, `bilinear=False`.

*   **Architecture Flow**:
    1.  **Input**: `(N, 25, H, W)`
    2.  **Inc**: `DoubleConv(25, 32)` processes the **Input** -> `inc`
    3.  **Encoder 1**: `Down(32, 64)` processes `inc` -> `encoder1`
    4.  **Encoder 2**: `Down(64, 128)` processes `encoder1` -> `encoder2`
    5.  **Encoder 3**: `Down(128, 256)` processes `encoder2` -> `encoder3`
    6.  **Bottleneck**: `Down(256, 512)` processes `encoder3` -> `bottleneck`
    7.  **Decoder 1**: `Up(512, 256, 256)` receives `bottleneck` and `encoder3` -> `decoder1`
    8.  **Decoder 2**: `Up(256, 128, 128)` receives `decoder1` and `encoder2` -> `decoder2`
    9.  **Decoder 3**: `Up(128, 64, 64)` receives `decoder2` and `encoder1` -> `decoder3`
    10. **Decoder 4**: `Up(64, 32, 32)` receives `decoder3` and `inc` -> `decoder4`
    11. **Output**: `OutConv(32, 5)` receives `decoder4` -> `final` (shape `(N, 5, H, W)`)

---

### **2. Unet-Cbam**

*   **Parameters**: `in_channels=25`, `initial_filters=32`, `num_classes=5`, `bilinear=False`.

*   **Architecture Flow**:
    1.  **Input**: `(N, 25, H, W)`
    2.  **Inc**: `DoubleConv(25, 32)` processes the **Input**, then through `CBAM(32)` -> `inc_cbam`
    3.  **Encoder 1**: `Down(32, 64)` processes `inc_cbam`, then through `CBAM(64)` -> `encoder1_cbam`
    4.  **Encoder 2**: `Down(64, 128)` processes `encoder1_cbam`, then through `CBAM(128)` -> `encoder2_cbam`
    5.  **Encoder 3**: `Down(128, 256)` processes `encoder2_cbam`, then through `CBAM(256)` -> `encoder3_cbam`
    6.  **Bottleneck**: `Down(256, 512)` processes `encoder3_cbam`, then through `CBAM(512)` -> `bottleneck_cbam`
    7.  **Decoder 1**: `Up(512, 256, 256)` receives `bottleneck_cbam` and `encoder3_cbam` -> `decoder1`
    8.  **Decoder 2**: `Up(256, 128, 128)` receives `decoder1` and `encoder2_cbam` -> `decoder2`
    9.  **Decoder 3**: `Up(128, 64, 64)` receives `decoder2` and `encoder1_cbam` -> `decoder3`
    10. **Decoder 4**: `Up(64, 32, 32)` receives `decoder3` and `inc_cbam` -> `decoder4`
    11. **Output**: `OutConv(32, 5)` receives `decoder4` -> `final` (shape `(N, 5, H, W)`)

---

### **3. Unet-BandSelector**

*   **Parameters**: `in_channels=25`, `reduced_depth=3`, `initial_filters=32`, `num_classes=5`, `bilinear=False`.

*   **Architecture Flow**:
    1.  **Input**: `(N, 25, H, W)`
    2.  **Preprocessing**: `BandSelector(25, 3)` processes the **Input** -> `preprocessed` (shape `(N, 3, H, W)`)
    3.  **Inc**: `DoubleConv(3, 32)` processes `preprocessed` -> `inc`
    4.  **Encoder 1**: `Down(32, 64)` processes `inc` -> `encoder1`
    5.  **Encoder 2**: `Down(64, 128)` processes `encoder1` -> `encoder2`
    6.  **Encoder 3**: `Down(128, 256)` processes `encoder2` -> `encoder3`
    7.  **Bottleneck**: `Down(256, 512)` processes `encoder3` -> `bottleneck`
    8.  **Decoder 1**: `Up(512, 256, 256)` receives `bottleneck` and `encoder3` -> `decoder1`
    9.  **Decoder 2**: `Up(256, 128, 128)` receives `decoder1` and `encoder2` -> `decoder2`
    10. **Decoder 3**: `Up(128, 64, 64)` receives `decoder2` and `encoder1` -> `decoder3`
    11. **Decoder 4**: `Up(64, 32, 32)` receives `decoder3` and `inc` -> `decoder4`
    12. **Output**: `OutConv(32, 5)` receives `decoder4` -> `final` (shape `(N, 5, H, W)`)

---

### **4. Unet-SEAttention (SAC)**

*   **Parameters**: `in_channels=25`, `reduced_depth=3`, `initial_filters=32`, `num_classes=5`, `bilinear=False`.

*   **Architecture Flow**:
    1.  **Input**: `(N, 25, H, W)`
    2.  **Preprocessing**: `SAC(in_depth=25, reduced_depth=3)` processes the **Input** -> `preprocessed` (shape `(N, 3, H, W)`)
    3.  **Inc**: `DoubleConv(3, 32)` processes `preprocessed` -> `inc`
    4.  **Encoder 1**: `Down(32, 64)` processes `inc` -> `encoder1`
    5.  **Encoder 2**: `Down(64, 128)` processes `encoder1` -> `encoder2`
    6.  **Encoder 3**: `Down(128, 256)` processes `encoder2` -> `encoder3`
    7.  **Bottleneck**: `Down(256, 512)` processes `encoder3` -> `bottleneck`
    8.  **Decoder 1**: `Up(512, 256, 256)` receives `bottleneck` and `encoder3` -> `decoder1`
    9.  **Decoder 2**: `Up(256, 128, 128)` receives `decoder1` and `encoder2` -> `decoder2`
    10. **Decoder 3**: `Up(128, 64, 64)` receives `decoder2` and `encoder1` -> `decoder3`
    11. **Decoder 4**: `Up(64, 32, 32)` receives `decoder3` and `inc` -> `decoder4`
    12. **Output**: `OutConv(32, 5)` receives `decoder4` -> `final` (shape `(N, 5, H, W)`)

---

### **5. Unet-Bcd**

This architecture uses a Band Compression Decompression (BCD) block as a feature refinement step on the input before it enters the standard `Unet-Base`. The BCD block compresses the channels and then immediately decompresses them, forcing the model to learn a compact representation.

*   **Parameters**: `in_channels=25`, `intermediate_channels=3`, `initial_filters=32`, `num_classes=5`, `bilinear=False`.

*   **Architecture Flow**:
    1.  **Input**: `(N, 25, H, W)`
    2.  **Preprocessing**: `BandCompressionDecompression(25, 3)` processes the **Input**. This involves a 1x1 Conv to 3 channels, a ReLU, and another 1x1 Conv back to 25 channels -> `preprocessed` (shape `(N, 25, H, W)`)
    3.  **Inc**: `DoubleConv(25, 32)` processes `preprocessed` -> `inc`
    4.  **Encoder 1**: `Down(32, 64)` processes `inc` -> `encoder1`
    5.  **Encoder 2**: `Down(64, 128)` processes `encoder1` -> `encoder2`
    6.  **Encoder 3**: `Down(128, 256)` processes `encoder2` -> `encoder3`
    7.  **Bottleneck**: `Down(256, 512)` processes `encoder3` -> `bottleneck`
    8.  **Decoder 1**: `Up(512, 256, 256)` receives `bottleneck` and `encoder3` -> `decoder1`
    9.  **Decoder 2**: `Up(256, 128, 128)` receives `decoder1` and `encoder2` -> `decoder2`
    10. **Decoder 3**: `Up(128, 64, 64)` receives `decoder2` and `encoder1` -> `decoder3`
    11. **Decoder 4**: `Up(64, 32, 32)` receives `decoder3` and `inc` -> `decoder4`
    12. **Output**: `OutConv(32, 5)` receives `decoder4` -> `final` (shape `(N, 5, H, W)`)

---

### **6. Unet-Cbam-Seattention**

This model combines two attention mechanisms. First, it uses the `SAC` block to perform spectral dimensionality reduction on the input. Then, the resulting tensor is fed into a `Unet-Base` architecture that has `CBAM` blocks integrated into its encoder path.

*   **Parameters**: `in_channels=25`, `reduced_depth=3`, `initial_filters=32`, `num_classes=5`, `bilinear=False`.

*   **Architecture Flow**:
    1.  **Input**: `(N, 25, H, W)`
    2.  **Preprocessing**: `SAC(in_depth=25, reduced_depth=3)` processes the **Input** -> `preprocessed` (shape `(N, 3, H, W)`)
    3.  **Inc**: `DoubleConv(3, 32)` processes `preprocessed`, then through `CBAM(32)` -> `inc_cbam`
    4.  **Encoder 1**: `Down(32, 64)` processes `inc_cbam`, then through `CBAM(64)` -> `encoder1_cbam`
    5.  **Encoder 2**: `Down(64, 128)` processes `encoder1_cbam`, then through `CBAM(128)` -> `encoder2_cbam`
    6.  **Encoder 3**: `Down(128, 256)` processes `encoder2_cbam`, then through `CBAM(256)` -> `encoder3_cbam`
    7.  **Bottleneck**: `Down(256, 512)` processes `encoder3_cbam`, then through `CBAM(512)` -> `bottleneck_cbam`
    8.  **Decoder 1**: `Up(512, 256, 256)` receives `bottleneck_cbam` and `encoder3_cbam` -> `decoder1`
    9.  **Decoder 2**: `Up(256, 128, 128)` receives `decoder1` and `encoder2_cbam` -> `decoder2`
    10. **Decoder 3**: `Up(128, 64, 64)` receives `decoder2` and `encoder1_cbam` -> `decoder3`
    11. **Decoder 4**: `Up(64, 32, 32)` receives `decoder3` and `inc_cbam` -> `decoder4`
    12. **Output**: `OutConv(32, 5)` receives `decoder4` -> `final` (shape `(N, 5, H, W)`)

---

### **7. Unet-DepthwiseSeparable**

This is a computationally efficient version of U-Net that replaces all standard `DoubleConv` blocks with `DoubleDSConv` blocks, which use depthwise separable convolutions. This significantly reduces the number of model parameters.

*   **Parameters**: `in_channels=25`, `initial_filters=32`, `num_classes=5`, `bilinear=False`.

*   **Architecture Flow**:
    1.  **Input**: `(N, 25, H, W)`
    2.  **Inc**: `DoubleDSConv(25, 32)` processes the **Input** -> `inc`
    3.  **Encoder 1**: `MaxPool2d` then `DoubleDSConv(32, 64)` processes `inc` -> `encoder1`
    4.  **Encoder 2**: `MaxPool2d` then `DoubleDSConv(64, 128)` processes `encoder1` -> `encoder2`
    5.  **Encoder 3**: `MaxPool2d` then `DoubleDSConv(128, 256)` processes `encoder2` -> `encoder3`
    6.  **Bottleneck**: `MaxPool2d` then `DoubleDSConv(256, 512)` processes `encoder3` -> `bottleneck`
    7.  **Decoder 1**: `Up` block using `DoubleDSConv` receives `bottleneck` and `encoder3` -> `decoder1`
    8.  **Decoder 2**: `Up` block using `DoubleDSConv` receives `decoder1` and `encoder2` -> `decoder2`
    9.  **Decoder 3**: `Up` block using `DoubleDSConv` receives `decoder2` and `encoder1` -> `decoder3`
    10. **Decoder 4**: `Up` block using `DoubleDSConv` receives `decoder3` and `inc` -> `decoder4`
    11. **Output**: `OutConv(32, 5)` receives `decoder4` -> `final` (shape `(N, 5, H, W)`)

---

### **8. Unet-DeepSupervision**

This architecture adds auxiliary outputs at different stages of the decoder. This provides extra supervision signals during training, which can help with convergence. The model returns a list of outputs, with the first being the main, full-resolution output.

*   **Parameters**: `in_channels=25`, `initial_filters=32`, `num_classes=5`, `depth=4`.

*   **Architecture Flow**:
    1.  **Input**: `(N, 25, H, W)`
    2.  **Inc**: `DoubleConv(25, 32)` -> `inc`
    3.  **Encoder 1**: `Down(32, 64)` -> `encoder1`
    4.  **Encoder 2**: `Down(64, 128)` -> `encoder2`
    5.  **Encoder 3**: `Down(128, 256)` -> `encoder3`
    6.  **Bottleneck**: `Down(256, 512)` -> `bottleneck`
    7.  **Decoder 1**: `Up(512, 256, 256)` receives `bottleneck` and `encoder3` -> `decoder1`
    8.  **Decoder 2**: `Up(256, 128, 128)` receives `decoder1` and `encoder2` -> `decoder2`
    9.  **Decoder 3**: `Up(128, 64, 64)` receives `decoder2` and `encoder1` -> `decoder3`
        *   **Auxiliary Output 1**: `OutConv(64, 5)` receives `decoder3` -> `ds_1` (shape `(N, 5, H/2, W/2)`)
    10. **Decoder 4**: `Up(64, 32, 32)` receives `decoder3` and `inc` -> `decoder4`
        *   **Auxiliary Output 2**: `OutConv(32, 5)` receives `decoder4` -> `ds_2` (shape `(N, 5, H, W)`)
    11. **Main Output**: `OutConv(32, 5)` receives `decoder4` -> `main` (shape `(N, 5, H, W)`)
    12. **Final Return**: The model returns a list: `[main, ds_2, ds_1]`