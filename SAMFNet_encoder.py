encoder:
  - [2, ConvNormLayer_fuse, [128, 1, 1]] 
  - [3, ConvNormLayer_fuse, [128, 1, 1]] 
  - [4, ConvNormLayer_fuse, [128, 1, 1]] 

  - [-1, (SEMFB, [128, {'module': GSA, 'param':{'num_heads': 8}}, {'module': FMFFN, 'param': {'window_size': 4, 'act_layer': nn.GELU}}, {'selfatt': True}]] 

  - [[8, 6, 5], TSFF, []] 
  - [-1, ConvNormLayer_fuse, [128, 3, 2]] 
  - [9, nn.Upsample, [None, 2, "nearest"]] 

  - [[10, 8], Concat, []] 
  - [-1, PHAM, [128, 256, 22, 2, False, 'silu']] 

  - [[11, 5], Concat, []] 
  - [-1, PHAM, [128, 256, 22, 2, False, 'silu']] 

  - [[13, 9, 15], TSFF, []] 

decoder:
  - [[15, 16], DFINETransformer, {"feat_strides":[8, 16], "hidden_dim": 128, "num_levels": 2, "num_layers": 3, "num_points": [6, 6], "dim_feedforward": 512}]