from fusion.quality_weight import AdaptiveQualityWeight

fusion = AdaptiveQualityWeight()

fusion.set_quality(

    eeg_quality=60.43,

    emg_quality=51.77

)

weights = fusion.compute()

fusion.summary()

print(weights)