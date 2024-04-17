## QLoRA's convention: 
- (a) 4-bit NormalFloat (NF4), a new data type that
is information theoretically optimal for normally distributed weights 
- (b) Double Quantization to reduce the average memory footprint by quantizing the quantization constants
- (c) Paged Optimizers to manage memory spikes. 