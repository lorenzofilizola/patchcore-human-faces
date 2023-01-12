datapath=/mnt/c/Datasets/patchcore_faces_eyes/eyes_cropped datasets=('faces')
dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '$dataset; done))


python src/run_patchcore.py --gpu 0 --seed 0 --save_patchcore_model --save_segmentation_images \
--log_group vgg19_eyes_cropped_with_flips --log_project test_Results results \
patch_core -b vgg19 -le features.30 --faiss_on_gpu \
--pretrain_embed_dimension 1024  --target_embed_dimension 1024 --anomaly_scorer_num_nn 1 --patchsize 3 \
sampler -p 0.1 approx_greedy_coreset dataset --resize=512 --imagesize=512 "${dataset_flags[@]}" flickrfaces $datapath
