from train import defocus_exp

# default parameters
if True:
    r = defocus_exp.run(config_updates={
        'TRAIN_PARAMS': {'ARCH_NUM': 1,
                        'EPOCHS_NUM': 2001,  'EPOCH_START': 0,
                        'FILTER_NUM': 16,
                        'RANDOM_LEN_INPUT': 0,

                        'TRAINING_MODE':2,

                        'MODEL1_LOAD': True,
                        'MODEL1_ARCH_NUM': 1,
                        'MODEL1_NAME': 'd02_t01',
                        'MODEL1_INPUT_NUM': 4,
                        'MODEL1_EPOCH': 1000, 'MODEL1_FILTER_NUM': 16,
                        'MODEL1_LOSS_WEIGHT': 1.,

                        'MODEL2_LOAD':False, 'MODEL2_NAME':'a44_d01_t01',
                        'MODEL2_EPOCH': 700,
                        },
        'DATA_PARAMS': {'DATA_NUM': 6,
                        'FLAG_SHUFFLE': False,
                        'INP_IMG_NUM': 5,
                        'FLAG_IO_DATA': {
                            'INP_RGB': True,
                            'INP_COC': False,
                            'INP_DIST': True,

                            'OUT_COC': True,
                            'OUT_DEPTH': True,
                        },
                        'BATCH_SIZE': 4,
                        'DATA_RATIO_STRATEGY': 0,
                        'FOCUS_DIST': [0.1,.15,.3,0.7,1.5],
                        'F_NUMBER': 1.,
                        },
        'OUTPUT_PARAMS': {
            'VIZ_HOSTNAME': "http://localhost", 'VIZ_ENV_NAME':'main',
            'COMMENT': "Test run",
            'EXP_NUM': 11,
        }
        })