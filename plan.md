# Ideas
1. added EMA for less variance 
2. Add cfg for more accurate diffusions
3. use https://github.com/konstmish/prodigy?tab=readme-ov-file as optimizer for good learning rate 



# Future Ideas

1. Increase model size drastically (ran 17h 5/19)

# What was done
1. create small dataset of 1000 paths on 1 world 
    * difficult to generalize
2. create big dataset on 1 world 
    * Did but model still bumped slightly into walls. did not train till the end
3. add conditioning based on world index
    * this conditioning had models do barely 0.2 accuracy 
4. add conditioning of vector fields
    a. tried using flatten embeder
        * results instantly 

