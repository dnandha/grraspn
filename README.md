# Train
`CUDA_VISIBLE_DEVICES="2,3" python tools/train_net.py --config-file configs/grasp_rcnn_R_50_FPN_3x.yaml --num-gpus 2`

Run training on TRAIN dataset specified in config.

# Validate
`CUDA_VISIBLE_DEVICES="2,3" python tools/train_net.py --config-file configs/grasp_rcnn_R_50_FPN_3x.yaml --num-gpus 2 --eval --resume`

Run validation on TEST dataset specified in config using latest trained model.

# Test
`CUDA_VISIBLE_DEVICES="2,3" python demo/demo.py --config-file configs/grasp_rcnn_R_50_FPN_3x.yaml --input <INPUT_IMAGE> --output <OUTDIR> --opts MODEL.WEIGHTS <MODELPATH>`

Save image with predicted grasps to given output directory.
