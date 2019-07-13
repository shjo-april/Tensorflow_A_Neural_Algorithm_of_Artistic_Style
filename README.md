# Tensorflow_A_Neural_Algorithm_of_Artistic_Style

## Run
```sh
python Train.py --content_path='./content_image/turtle.jpg'
                --style_path='./style_image/kadinsky.jpg'
                --save_dir='./results_turtle_kadinsky/'
```

## Results (turtle + kadinsky)
![result](./res/result_turtle.gif)

## Results (My Cat + kadinsky)
![result](./res/result_SSAL_kadinsky.gif)

## Results (My Cat + Gogh)
![result](./res/result_SSAL_Gogh.gif)

## Requirements
- Tensorflow 1.13.1
- OpenCV 4.0.0
- Numpy 1.16.4
- http://download.tensorflow.org/models/vgg_19_2016_08_28.tar.gz