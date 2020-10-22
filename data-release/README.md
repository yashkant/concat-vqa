# Data Setup
We use a publicly available detection model `vqa-maskrcnn-benchmark` to extract image features. 
But you don't need to do that!? Just use this [dropbox link](https://www.dropbox.com/sh/v826l3ge7oz4vz4/AACRimDdy_BGnN2XZJDZLyY6a?dl=0) to download everything (features, splits, negatives). 
If you insist on extracting features see [this section](##Extracting features).


## Data Organization
Organize all the data files as per below structure:
```
data-release
|
├── image-features/
│   ├── COCO_test_resnext152_faster_rcnn_genome.lmdb        # extracted features from train-val
│   └── COCO_trainval_resnext152_faster_rcnn_genome.lmdb    # extracted features from test         
|
├── negative-files/
│   ├── fil_trainval_quesiton_negs.pkl
│   ├── fil_val_quesiton_negs.pkl
│   └── fil_train_question_negs.pkl
|
├── splits/
|    └── <lots of files (17 to be precise) here, copy all from dropbox>
|
├── README.md
└── extract_features.py
```



## Extracting features

Install `vqa-maskrcnn-benchmark`([link](https://gitlab.com/vedanuj/vqa-maskrcnn-benchmark)) repository and download the model and config. 

```text
cd data-release
wget https://dl.fbaipublicfiles.com/vilbert-multi-task/detectron_model.pth
wget https://dl.fbaipublicfiles.com/vilbert-multi-task/detectron_config.yaml
```


To extract features for images, run from root directory:

```text
python data-release/extract_features.py --model_file data/detectron_model.pth --config_file data/detectron_config.yaml --image_dir <path_to_directory_with_images> --output_folder <path_to_output_extracted_features>
```

#### Extract features for images with Ground Truth bboxes

Generate a `.npy` file with the following format for all the images and their bboxes

```text
{
    {
        'file_name': 'name_of_image_file',
        'file_path': '<path_to_image_file_on_your_disk>',
        'bbox': array([
                        [ x1, y1, width1, height1],
                        [ x2, y2, width2, height2],
                        ...
                    ]),
        'num_box': 2
    },
    ....
}
```

Run from root directory

```text
python data-release/extract_features.py --model_file data/detectron_model.pth --config_file data/detectron_config.yaml --imdb_gt_file <path_to_imdb_npy_file_generated_above> --output_folder <path_to_output_extracted_features>
```

Convert the extracted images to an LMDB file

```text
python data-release/convert_to_lmdb.py --features_dir <path_to_extracted_features> --lmdb_file <path_to_output_lmdb_file>
```

