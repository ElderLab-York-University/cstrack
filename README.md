# CSTrack

CSTrack proposes a strong ReID based one-shot MOT framework. It includes a novel cross-correlation network that can effectively impel the separate branches to learn task-dependent representations, and a scale-aware attention network that learns discriminative embeddings to improve the ReID capability. This makes the data association ability of the model comparable to two-stage methods, while running faster.

<img src="https://raw.githubusercontent.com/ElderLab-York-University/cstrack/master/demo/CSTrack_CCN.jpg" height="300" alt="CSTrack"/><br/>

## Testing

Download the [pre-trained model](https://yuoffice.sharepoint.com/:u:/s/LE-SENTRYnet/ERMGgpNYk6FPgeEBu_QScP8BWn8YIkby_WSuIJQSVmy1Yw?e=kn1G71) and save it to `CSTrack/weights`.

Download the [MOT-16](https://motchallenge.net/data/MOT16/) dataset and save it to `CSTrack/dataset`.

Open a prompt to `CSTrack/tracking` and enter the command below:

    python test_cstrack.py --test_mot16 True

## See Also

* [Paper](https://arxiv.org/abs/2010.12138)
* [Original GitHub repository](https://github.com/JudasDie/SOTS/tree/MOT)
* [Demo](https://motchallenge.net/method/MOT=3601&chl=10)
