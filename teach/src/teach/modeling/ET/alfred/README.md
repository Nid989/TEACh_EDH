# Files Structure

```
/data
    create_lmdb.py       (script to create an LMDB dataset out of trajectory files)
    preprocessor.py      (class to preprocess trajectories annotations and actions)
    preprocessor_et.py   (class to preprocess trajectories annotations and actions for original ET model)
    zoo/base.py          (base class for LMDB dataset loading using multiple threads
/model
    train.py             (script for models training)
    base.py              (base class for E.T. and translator models)
    learned.py           (class with main train routines)
    speaker.py           (translator model)
    transformer.py       (E.T. model)
/nn
    attention.py         (basic attention mechanisms)
    dec_object.py        (object decoder class)
    enc_lang.py          (language encoder class)
    enc_visual.py        (visual observations encoder class)
    enc_vl.py            (multimodal encoder class)
    encodings.py         (positional and temporal encodings)
    transforms.py        (visual observations transformations)
/utils
    data_util.py         (data handling utils)
    eval_util.py         (evaluation utils)
    helper_util.py       (help utils)
    metric_util.py       (utils to compute scores)
    model_util.py        (utils for E.T. and translation models)
```
