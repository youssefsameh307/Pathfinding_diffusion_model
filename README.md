# RobotPathData

Collection of paths for different robots and environments.

Until a better solution with GitHub is found, the data itself lies on GoogleDrive.


---
# Dependencies
* [wzk](https://github.com/scleronomic/WerkZeugKasten):   Note this is a different package from pip install wzk
1. Add public key to github 1st
`pip install git+ssh://git@github.com/scleronomic/wzk.git@stable-adlr`


2. Get Data
import gdown
gdown.download_folder("https://drive.google.com/drive/folders/16UthOas97BntH8ngVSofxgd2JrxSLDZU", quiet=True)

