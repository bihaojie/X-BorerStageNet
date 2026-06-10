backbone:
  - [-1, APSDM, [16, 16, True]] 
  - [-1, DSHGB, [16, 64, 1, 3, False, False, 3, True, 'se']] 
  - [-1, DSHGB, [32, 256, 1, 3, True, False, 3, True, 'se']] 
  - [-1, DSHGB, [64, 512, 2, 3, True, True, 5, True, 'se']]  
  - [-1, DSHGB, [128, 1024, 1, 3, True, True, 5, True, 'se']] 
