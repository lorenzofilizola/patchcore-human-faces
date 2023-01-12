datapath=/mnt/c/Datasets/patchcore_faces_eyes/eyes_cropped datasets=('faces')
loadpath='results/test_Results'

modelfolder=vgg19_eyes_cropped_14
savefolder=evaluated_results'/'$modelfolder'_test'

model_flags=($(for dataset in "${datasets[@]}"; do echo '-p '$loadpath'/'$modelfolder'/models/flickrfaces_faces'; done))
dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '$dataset; done))

python src/load_and_evaluate_patchcore.py --save_segmentation_images --gpu 0 --seed 0 $savefolder \
patch_core_loader "${model_flags[@]}" --faiss_on_gpu \
dataset --resize 512 --imagesize 512 "${dataset_flags[@]}" flickrfaces $datapath