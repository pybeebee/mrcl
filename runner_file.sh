python mrcl_regression.py --update_step 40 --meta_lr 0.0001 --update_lr 0.003 --tasks 10 --capacity 10 --width 300 --rln 6

python mrcl_regression_L1Loss.py --update_step 40 --meta_lr 0.0001 --update_lr 0.003 --tasks 10 --capacity 10 --width 300 --rln 6

python mrcl_regression_freezing_control.py --update_step 40 --meta_lr 0.0001 --update_lr 0.003 --tasks 10 --capacity 10 --width 300 --rln 6

# python mrcl_classification.py --rln 6 --update_lr 0.03 --name mrcl_omniglot --update_step 20 --steps 15000

# python mrcl_classification_hinge.py --rln 6 --update_lr 0.03 --name mrcl_omniglot --update_step 20 --steps 15000

# python mrcl_classification_freezing_control.py --rln 6 --update_lr 0.03 --name mrcl_omniglot --update_step 20 --steps 15000
