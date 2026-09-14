# observations.md

## Pahse 6 - New dataset config

### Gated Attention (168 attentiondim)
- val loss min occoured at 45 epochs, trigerring best model there. But test accuracy is better at the final model after 100 epochs, even though val loss is higher

- now normally, one would want to keep training to get the best possible performance on the val set. But that is when your target test set is the same distribution as the val set.
- In our case, we have 2 test sets - one is the same distribution as the val set. The other is a test to see how well the model generalizes to the same classes in different settings. But even in the 2nd test set, the features that are important should be the same... 
- But from the limited experiments we have done, it seems that over-optimizing on the val set can lead to worse performance on the test set...
- So, now the question is, how do we know when to stop training?

    ### Why we need a OOD val set
    - It was observed that models that were checkpointed at earlier epochs, before the target validation loss reached a minimum, performed better on the hand (ood) test set.
    - Our objective is to develop a model that generalizes well to images in different settings, not just the primary distribution set.
    - Therefore, we need to include an out-of-distribution (OOD) validation set to identify the best stopping point for training.

    ## Phase 7 - Validation OOD included to identify best stopping point

    ### Best Model selection options:

    1. Save all promising models
        a. Best target val_loss model
        b. Best ood val_loss model
        c. Best target val_acc model
        d. Best ood val_acc model
        e. where ood val_loss starts to increase (both are not improving together anymore)

    And then we can compare and choose a suitable model.
    # BEST MODEL SELECTION
        For now, we are saving all promising models
          - best losses and accuracies for val and ood sets
        That will make 4 + 1 models saved at the end of training for each run.
        The Patience thing is only with the in-domain validation set - and early stopping is disabled for now.
        Of these, the most promising should be the one with best ood val loss or maybe accuracy - is what i think.