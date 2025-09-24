.PHONY: data

PY=python

data:
	$(PY) scripts/prepare_data.py --root /Users/boudoul/Dev/dinov3-isic2018-segmentation/data/isic2018 --splits train val test 

