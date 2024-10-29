# CLIP training
Code to train [CLIP](https://github.com/openai/CLIP) on [MS-COCO](https://cocodataset.org/#home) captions. 

Github: [https://github.com/revantteotia/clip-training]

**Dataset Details**
  -  Dataset: MS-COCO 
  -  Number of samples: 118k
**Model Details**
  -  Image encoder: ResNet50
  -  image_resolution : 224 
  -  context_length : 77
  -  vocab_size : 49408
  -  transformer_width : 512
  -  transformer_heads : 8
  -  transformer_layers : 6 # 12 in CLIP
  -  Number OF PARAMETERS:  83,092,833

**Training Details**
-  Trained on ARC
-  Duration: Approximately took 4 hours to complete
-  Epoch 35
-  Batch size: 64 per GPU

![image](https://github.com/user-attachments/assets/8f7621e4-2138-43f6-9d8d-685cdb7165ec)


![image](https://github.com/user-attachments/assets/73fcd252-7912-4f93-8051-9a1b16e4e819)
