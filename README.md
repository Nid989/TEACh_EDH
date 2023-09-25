# PM:LVI-23 TEACh
This repository consists of the work built up on the Execution from Dialog History task mentioned in [Task-driven Embodied Agents that Chat](https://arxiv.org/abs/2110.00534). This work is done as a Project Module - Language Vision and Interaction at the University of Potsdam (SoSe 23) by Nidhir Bhavsar and Kushal Koshti under the supervision og Prof. David Schlangen

## 1) Execution from Dialog History (EDH) 

### Pre-requisites
- python3 `>=3.7,<=3.8`
- os: `Linux, Mac`
- python3.x-dev, example: `sudo apt install python3.8-dev`
- tmux, example: `sudo apt install tmux`
- xorg, example: `sudo apt install xorg openbox`
- ffmpeg, example: `sudo apt install ffmpeg`

### Installation
```
pip install -r teach/requirements.txt
pip install -e teach/.
```

### Download the dataset
```
teach_download -d teach_data 
```
This will download and extract all the files (`experiment_games.tar.gz`, `all_games.tar.gz`, 
`images_and_states.tar.gz`, `edh_instances.tar.gz` & `tfd_instances.tar.gz`) in 'TEACH_PME/teach_data' directory. Change the argument -d as required.
**Arguments (optional):**
- `-d`/`directory`: The location to store the dataset into. Default=`/tmp/teach-dataset`.
- `-se`/`--skip-extract`: If set, skip extracting archive files.
- `-sd`/`--skip-download`: If set, skip downloading archive files.
- `-f`/`--file`: Specify the file name to be retrieved from S3 bucket.

### Remote Server setup 
This setup is required when doing Inference on a remote cluster without any display. If you only need to train the model and move the inference to a different machine with display, this step can be skipped.

Start an X-server 
```
tmux
sudo python ./bin/startx.py
```
Exit the `tmux` session (`CTRL+B, D`). Any other commands should be run in the main terminal / different sessions.

### Create LMDB dataset from TEACh data 

```
export ET_ROOT=./teach/src/teach/modeling/ET
export ET_LOGS=./teach/generations/sample_model_et
export TEACH_DIR=./teach
export PYTHONPATH=$TEACH_DIR/src/teach/modeling/ET:$PYTHONPATH
export ET_DATA=./teach_data  
(Assuming the dataset was stored in 'TEACh_PME/teach_data', as shown before)

python3 teach/src/teach/modeling/ET/alfred/data/create_lmdb.py

```
This will create the dataset for train, valid_seen and valid_unseen in the 'TEACh_PME/teach_data' folder. The `worker00` folder can be deleted if storage becomes an issue later on. Various configurations can be set in `create_lmdb.py` while creating the dataset such as creating a smaller dataset (fast_epoch) or selecting the visual architecture type (visual_archi). Set the argument (original_et) to create the dataset for training the baseline E.T. models

### Train the model on created LMDB dataset

```
export ET_ROOT=./teach/src/teach/modeling/ET
export ET_LOGS=./path/to/save/log/files
export TEACH_DIR=./teach
export PYTHONPATH=$TEACH_DIR/src/teach/modeling/ET:$PYTHONPATH
export ET_DATA=./teach_data  
(Assuming the dataset was stored in 'TEACh_PME/teach_data', as shown before)

python3 teach/src/teach/modeling/ET/alfred/model/train.py

```

Various configurations mentioned in our paper can be setup in `model/config.py` for training. Gather the following files [`config.json`, `params.json`, and `checkpoint.pth` (Checkpoint of the model you want to inference on -> Should be saved as `latest.pth`) ] and save in a new baseline folder (This will be used in next step for infernece). To this folder, add `data.vocab` and `params.json` file which were created during previous step (Creating LMDB dataset).


### Evaluate the models

```
export ET_ROOT=./teach/src/teach/modeling/ET
export ET_LOGS=./path/to/save/log/files
export ET_DATA=./teach_data
export TEACH_DIR=./teach
export PYTHONPATH=$TEACH_DIR/src/teach/modeling/ET:$PYTHONPATH
export DATA_DIR=./teach_data

export OUTPUT_DIR=./path/to/save/output/predicted_actions
export METRICS_FILE=./path/to/save/metrics/without/extension
export IMAGE_DIR=./path/to/store/images

teach_inference \
    --data_dir $DATA_DIR \
    --images_dir $IMAGE_DIR \
    --split valid_seen \
    --model_module teach.inference.et_model \
    --model_class ETModel \
    --model_dir ./path/to/folder/containing/new_model \
    --visual_checkpoint ./teach_data/et_pretrained_models/fasterrcnn_model.pth \
    --object_predictor ./teach_data/et_pretrained_models/maskrcnn_model.pth \
    --output_dir $OUTPUT_DIR \
    --metrics_file $METRICS_FILE \
    --seed 1

```

 To calculate metrics while at the end of inference automatically, ensure that `./outputs/outputs` path exists (Assuming you've set `OUTPUT_DIR=./outputs`). If not, the subfolder should be manually created. Predicted actions and metric files will be stored at path './outputs/' and all inference files for each EDH instance will be stored at path './outputs/outputs/'. 


## 2) Finetuning Models for EDH Plan Prediction

We have included the notebooks as well as the base data that can be used to fine tune the models. The datasets comprose of three configurations:

- Dialog History Only (DH)
- Dialog History + Dialog Act Information (DH + DA)
- Dialog History + Dialog Act + Filter (DH + DA + F)

We have also included our inference results on Finetuned llama for DH dataset.