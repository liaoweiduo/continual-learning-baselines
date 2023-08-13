# To Reviewer xdij

We sincerely appreciate your constructive comments on this paper. We detail our response below point by point. Please kindly let us know if our response addresses the issues you raised in this paper. 

Q1: Differences between CFST and multi-label classification tasks

- We are very sorry that the benchmark construction process confuse you. We would like to highlight the difference between our CFST and mutli-label recognition task, which is illustrated in Figure 1 and Remark 3.2 [line 106-111] of main paper.
- We do not provide concept labels to the model, which is a more diffcult setting than multi-label classification tasks.
- As a result, the model has to learn the hidden concepts from the image itself.
- We also discussed the difficulties for models to learn compositionality in this way in Appendix A. Briefly speaking:
    1. Some beneficial features may be good for the specific classification task, but not good for understanding this class (in a compositional way). 
    2. It is hard for the model to tell which visual features capture the corresponding concepts. 
- Also, we are very sorry for not presenting the details of the construction process in the main paper due to page limit. Instead, we present it in Appendix C.1.1 and C.2.1.

Q3: This combination dataset does not meet the requirements of real-world scenarios and the task of compositional generalization

- Our CGQA is like you said constructed in a grid way. We claim that this is because it can provide human interpretable concept visualization and easy to analyse and diagnose the model’s compositionality.
- And we also provide COBJ, which is a real-world benchmark, which is not constructed in a grid way (detailed in Appendix C). We provide some image instances in Appendix Fig5 and the construction details in Appendix C.2.

Q4: Frozen modules limits the adaptability of the methods to new tasks

- I am not sure if I capture your question correctly, if you ask why those modularity-based methods (e.g., RPSnet) freeze old modules. They freeze old modules to address catastrophic forgetting on old tasks. As for adaptability to new tasks, they learn new modules sub-linearly to solve new knowledge. Since the old modules are fozen, the number of trainable parameters for each task is kept the same.
- In order to answer you concern, we also run experiments that the modules for old tasks are not frozen, thus, we jointly train all used modules in RPSnet on CGQA.
    - Results: Hn=67.99 (main paper frozen old modules’ Hn=59.94).
    - Jointly training can indeed improve performance. However, we should emphasize that this performance gain is due to the larger model capacity since the number of trainable parameters are relatively larger than that in other baselines which is unfair for comparison.

Q5: No discussion about the negative societal impact

- Thank you very much to point out the limitations. Our benchmarks do not contain any sensitive data, such as medical data, which makes them ethically sound sound for use in research.
- We will update the discussion in our new revision as soon as possible.

Thank you again for your comments.

# To Reviewer RpQ2

We sincerely appreciate your constructive comments on this paper. We detail our response below point by point. Please kindly let us know if our response addresses the issues you raised in this paper.

Q1: Model size comparison for RPSnet to other CL method. 

- Very good concern. RPSnet (also MNTDP) exactly has much more model parameters than other methods. However, RPSnet only allows one path of modules to train. That is, each layer has only one trainable module for one task. Overall, the number of trainable parameters are kept the same with other methods.
- We can not fix the model size for modularity-based methods since this is their key point to address catastrophic forgetting by sub-linearly increasing model capacity. What we can do to support fair comparison is to use the same number of trainable parameters in one task.

Q2: Limitation on selected concepts. 

- Very good comment and advices. It is exactly our limination. Note that, we want to achieve **balance** and **flexible** combinations of concepts. That is, the number of instances for different combinations of concepts can be similar (no long-tailed combinations) and we can combine any pair of concepts together. However, some concepts like motion and human mood can only be combined with the human concept (maybe it is better to explain as ``attribute'' in this work). This limits the flexibility and limits the number of re-combinations in the systematicity test since it is difficult to find various kinds of moods on other animals (maybe cats can :p ).
- In the future, we tend to study concepts in a hierarchical way. We can contain more ``abstract'' (not visual) concepts like you said, e.g., specifically combining facial or gesture benchmarks.

Thank you again for your comments.

# To Reviewer 9Qe4

We sincerely appreciate your constructive comments on this paper. We detail our response below point by point. Please kindly let us know if our response addresses the issues you raised in this paper.

Q1: Move related work from appendix to main paper

- Good advice. Unfortunately due to page limit, we do not include them in our main paper. However, we clearly understand that they are important for the reader so we will include them in the camera-ready version.

Q2: Too strict requirement of frozen feature extractor

- Very good comments. First I would like to highlight that our benchmarks do not impose any restriction on the methods. During CL training, methods’ feature extractor does not have to be frozen (which is also the case in our experiments). Maybe our claim in the main paper misleads you, and we will correct the expression in the revision.
- We now explain why we chose to use fewshot tasks and frozen feature extractor during evaluation. At specific checkpoint (after finishing all continual training tasks), we use our evaluation protocol to evaluate the model’s compositionality. In such condition, if the number of support samples in the evaluation task is large, the feature extractor may learn from them and thus we can not actually judge whether the good performance comes from the original feature extractor (learned from old tasks).
- We also try to not freezen the feature extractor. And the results are present in Appendix E.7. For your convenience, we show our observation: all methods show a performance drop if not freezing the feature extractor especially for ER. It is clearly an overfitting issue and the bad effect is method-dependent. So in order to eliminate this effect when comparing the accuracies among methods, we freeze the feature extractor.
- Note that methods which don't explicitly separate feature learning from classifier learning can also use our evaluation protocol, as long as they can handle few-shot cases.
    - [We provide a version with enough examples and not freezing the feature extractor during evaluation. In this case, the evaluation task can be treated as the N+1-th continual task. The results are shown below: ??感觉上面的claim也指出学FE不好，针对feature learning和classifier learning分不开的mothod，可以直接在few shot task上测，不freeze FE。 ]

Q3: ``Never-ending'' CL sense

- This is a very good comment. Our reported results are indeed static (only evaluating after finishing all continual training tasks), but the setting is clearly a never-ending CL.
- We claim that our diagnosing evaluation can be used at any checkpoint (not be restricted at the end of all continual training tasks). We report the per-task Hn on 10-way COBJ as following:
    
    ```markdown
    | | Hn_1 | Hn_2 | Hn_3 |
    | Finetune | 37.57 | 38.32 | 37.82 |
    |ER | 30.27 | 33.33 | 36.99 |
    |GEM | 38.05 | 37.60 | 37.36 |
    |LwF | 38.13 | 43.34 | 45.19 |
    |EWC | 37.34 | 36.86 | 38.68 |
    | Finetune* |37.41 | 37.07 | 40.93 | 
    | ER* | 31.11 | 37.30 | 38.66 |
    |GEM* | 37.45 | 35.60 | 40.93 |
    |LwF* | 37.84 | 44.01 | 44.75 |
    |EWC* | 30.76 | 34.18 | 36.20 |
    ```
    
    - where Hn_1 denotes Hn after finish the first continual task. We deepcopy the feature extractor and train a new classifier for each compositional testing task.
    - A rough observation: [pending for discussion]
- [想一个场景。医院里，[应该不用再举例了吧, 他说if not再举例]]

Q4: Short and superficial discussion of limitations of this work

- We are very sorry that due to page limit we do not discuss more on our limitation.
- Here we discuss more on one limitation: our candidate concepts were quite visual and disentangled. Note that, we want to achieve **balance** and **flexible** combinations of concepts. That is, the number of instances for different combinations of concepts can be similar (no long-tailed combinations) and we can combine any pair of concepts together. However, some concepts like motion and human mood can only be combined with the human concept (maybe it is better to explain as ``attribute'' in this work). This limits the flexibility and limits the number of re-combinations in the systematicity test since it is difficult to find various kinds of moods on other animals (maybe cats can :p ).

Q5: Relation To Prior Work

- I am not sure if I capture your question correctly, if your ``Prior Work'' refers to those benchmarking works since our main contributions are the benchmarks and the corresponding evaluation protocol. We mentioned in Sec 2 (Related Works) that some vision benchmarks evaluates compositionality (including works in the CL field) and we pointed out that some benchmarks are toy and evaluate only systematicity. Thus, we provided the diagnosing benchmark (CGQA) and real-world benchmark (COBJ), and evaluated three aspects of compsoitionality (sys, pro, sub).
- As for other comparison, unfortunately due to page limit, we did not include the detail discussion in our main paper. However, we clearly understand that they are important and will include them in the camera-ready version.
    - Additionally as for the relationship with CZSL, we illustrated the difference in Figure 1 and Remark 3.2.
    - As for the relationship with forgetting, we investigated empirically by a case study in Sec 6 [line 276-304] and also provided detail experiments in Appendix E.4.

Q6: CGQA disentangled concept feature  (Sec 3)

- See Q4.

Q7: Question about non-novel testing accuracy vs training accuracy in Sec 4

- Good question. The answer is yes, exactly as you think. Although one non-novel testing task contains the same number of labels as training tasks, the K labels are randomly chosen from the training label pool. That is, it is a small probability that a non-novel testing task is just one of the training task (of course, the number of training samples for each label is different from the training task).

Q8: Put more description of the construction process rather than the motivation in Sec 5

- Good comment. We will compress the motivation and put the construction process to the main paper in the camera-ready version.

Q9: Question about experimental results

1. Multitask is not the best on Hn in COBJ (RPSnet and ER* are the best)
    - Good question. Firstly, we would like to highlight that Hn evaluates compositionality of the feature extractor.
    - This observation is quite interesting, that, Multitask may not necessarily be the upper bound w.r.t. compositionality. You know in CGQA, compositionality is easier to learn since we visually split the concepts which the models are expected to learn. Multitask shows a great superiority on Hn in CGQA, which is also consistent with the CAM visualization results in Appendix E.3 and fig7. Multitask can recognize more concepts than Finetune.
    - However, in the real-world case COBJ, concepts are not as visually separable as that in CGQA. Our CAM visualization results in Appendix fig8 shows that Multitask is better than Finetune but the gap is not so large(the number of recognized concepts by Multitask is larger but similar than that by Finetune). That is, Multitask may not necessarily beats CL methods w.r.t. compositionality.
    - Hope my analysis solves your question.
2. MNTDP* is not the top performer on Hn on COBJ, only on Acon.
    - Good question. As we highlighted on the above question, a model with better Acon does not necessarily have better Hn (compositionality).
    - This result shows that MNTDP* shows no superiority on compositionality. The high average test accuracy is due to the zero forgetting of old tasks, since it freezes all learned modules for old tasks.
- The above two observations also indicates the shortcomings of average test accuracy Acon. Our evaluation metric Hn provides more insights.

Q10: Grammar errors and typos

- Sorry for my grammar errors and typos that lead to the misunderstanding of some parts.
- We will carefully check those typos in our revision.

Thank you again for your comments.

# To Reviewer fvLh

We sincerely appreciate your constructive comments on this paper. We detail our response below point by point. Please kindly let us know if our response addresses the issues you raised in this paper.

Q1: Contradiction between experimental results and the claim: "compositionality addresses the stability-plasticity dilemma" 

- Very good question. First we would like to highlight that we are discussing the stability and plasticity on the **feature extractor** of a continual learner. We can easily observe forgetting phenomenon that this continual learner has a performance drop on old tasks after learning a new task. However, we want to say that the performance drop is from two aspects.
    1. The feature extractor forgets the crucial features for old tasks when learning the new task. 
    2. The classifiers of old tasks can not update (no old sample), thus, the coupling between the feature extractor and the corresponding classifiers is broken. 
- In our experiment, we visualized CAM of feature extractor in Appendix E.3. On CGQA, the learned feature extractor of Finetune was compositional, thus, it had good stability. However, Acon was very bad, thus, Acon can not one-hundred percent represent the performance of plasticity and stability abilities.
- While, our Hn evaluate the compositionality of a feature extractor (how well can this feature extractor extract compositional features). We eliminate the effect of the classifier.
- By the way, our evaluation method is flexible and can also be used on algorithms which don't explicitly separate feature learning from classifier learning. We just provide few-shot testing tasks and algorithms can just deepcopy and evaluate their models with their own methods.
- 抛开modularity结果，baselines是基本上Hn高的Acon就好（table 1的2的ER，ER*） why ER better than LwF。
    - LwF的问题是class IL上的prediction bias on classifier较为严重，而memory-based方法ER却能很好的解决这个，他们的feature extractor在extractor.
- 而 modularity （RPSnet）在class-IL上好的原因是it reduces the prediction bias in the classifier (Appendix line 521-523), and our compositionalty more focuses on feature extractor (to extract compositional features).

Q2: Motivations on the evaluations of three compositinoal capabilities

- Very good question. As we claim and explain in the above question, compositionality is a very important ability for a continual learner. and most of the current works (Sec 2 related works) in vision only consider systematicity (novel re-combination) as composistionality. we extend to productivity and substitutivity (other two very interesting aspects of compositionality which are widely studied in the NLP field) to provide more insights of continual learners.
- We now discuss more about the motivations on productivity and substitutivity:
    - Productivity: Compositional feature extractors trained with simple combinations of concepts can easily generalize to complex images (more visual concepts). For example in our main paper line 150-151, a un-compositional feature extractor may learn coupled features between Grass and Table concepts when recognizing {Grass, Table}. Then, when seeing {Door, Leaves, Shirt, Table} (one image with the Table concept but no Grass concept), it does not have high activating values on these features. On the other hand, a compositional feature extractor learns decoupled features for Grass and for Table concepts. Thus, it can have higher activating values on Table features when seeing {Door, Leaves, Shirt, Table}. Productivity test is to evaluate this performance.
    - Substitutivity: In order to achieve **balance** and **flexible** combinations of concepts (the number of instances for different combinations of concepts can be similar (no long-tailed combinations) and we can combine any pair of concepts together), our selected concepts are all visual and disentangled. However, some concepts (like motion and human mood) are more likely to be the attribute of other concepts (like human). These attribute-like concepts are not so flexible that it can not appear on other concepts or the representation may be different when combined with different concepts (like Ripe apple is different from Ripe banana). To compensate the evaluation on these attribute-like concepts, we design substitutivity test.

Q3: Knowledge leaking on continual training tasks

- reviewer觉得CL setting不同task之间需要完全无关，novel classes没有knowledge leaking。我们的不同task会有相同的concept出现，不是standard CL setting。
- besides， we also evaluate on noc, (non-compositional testing), the concepts are all unseen on old tasks.

Q3: Explain why Principle 2 (few-shot learning) and Principle 3 (frozen feature extractor) supports evaluating model's compositionality.

- At specific checkpoint (after finishing all continual training tasks), we use our evaluation protocol to evaluate the model’s compositionality. In such condition, if the number of support samples in the evaluation task is large, the feature extractor may learn from them and thus we can not actually judge whether the good performance comes from the original feature extractor (learned from old tasks). Thus, we recommend few-shot evaluation tasks and frozen feature extractors. We also list the reason in the main paper line 120-126.
- We should highlight that the principles are not strict and we also did experiments on not frozen feature extractors in Appendix E.7. For your convenience, we show our observation: all methods show a performance drop if not freezing the feature extractor especially for ER. It is clearly an overfitting issue and the bad effect is method-dependent. So in order to eliminate this effect when comparing the methods and let the accuracy correctly represents the compositionality, we freeze the feature extractor.

Q4: Add literature review of more recent works 

- Thank you very much for providing recent works. I will put them into our related works. We will submit our revision as soon as possible.
- cvpr 2023 6月18才开，他让我加这个？？github code都没提供, 而且还是vision-language benchmark。can be a good candidate但是现在只能在related works里。
- DualPrompt和l2p还有****S-Prompts****是pretrained vit，我们claim that pretrained会potentially see test labels， unfair，所以我们实验都是from scratch，没有pretraining。吴槟那有跑dualprompt和l2p，似乎pretrain的comp test很差，可以展示一下
    - prompt本身的设计理念就是为了extract knowledge，跟vit一起去co-train，估计效果也不太好。

Q5: Provide more justification on the claim "forgetting is not as suffered as that in the class-IL setting on CGQA (Line 244-245)".

- Sorry, my wrong grammar leads to the misunderstanding. The correct claim is that “ This is because forgetting on CGQA is not as suffered as that on COBJ, especially in the task-IL setting. ” We will update this in our revision.
- To justify this, we show the test accuracies for Finetune just after finishing each continual training task as follows:
    - task-IL 10-way CGQA tasks
        
        ```markdown
        | finish task 1| 58.4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
        | 2 |55.2 | 66.7 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
        | 3 | 55.1 | 65.4 | 74.4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
        | 4 | 54.2 | 63.3 | 61.3 | 83.8 | 0 | 0 | 0 | 0 | 0 | 0 |
        | 5 | 54.7 | 56.4 | 59.4 | 71.9 | 75.4 | 0 | 0 | 0 | 0 | 0 |
        | 6 | 48.9 | 57.7 | 66.0 | 71.1 | 72.2 | 77.1 | 0 | 0 | 0 | 0 |
        | 7 | 54.0 | 58.7 | 63.8 | 75.9 | 66.6 | 72.4 | 75.3 | 0 | 0 | 0 |
        | 8 | 47.2 | 58.4 | 54.1 | 74.3 | 64.9 | 71.1 | 71.0 | 74.8 | 0 | 0 |
        | 9 | 61.6 | 70.0 | 68.2 | 82.9 | 76.4 | 76.4 | 75.5 | 69.9 | 84.8 | 0 |
        | 10 | 53.4 | 69.1 | 74.3 | 78.3 | 75.5 | 75.5 | 70.1 | 67.1 | 82.5 | 82.2 |
        |  | evaluate on task 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
        ```
        
    - task-IL 10-way COBJ tasks
        
        ```markdown
        | finish task 1| 54.1 | 0 | 0 |
        | 2 |28.1 | 55.1 | 0 | 
        | 3 | 35.5 | 29.5 | 53.3 | 
        |  | evaluate on task 1 | 2 | 3 |
        ```
        
- On CGQA, forgetting is relatively smaller than COBJ even for the naive Finetune method. It is intuitive, since COBJ is a real-world benchmark and CGQA is a grid-like synthesized benchmark.
- The key point of MNTDP* to address catastrophic forgetting is to freeze old modules, thus, it can achieve no forgeting. However, in our CGQA case, forgetting is not as suffered as the real-world benchmark (e.g., COBJ). Thus, MNTDP* does not perform well on CGQA. But, in COBJ, MNTDP* can largely eliminate forgetting, thus, outperforms the others.

Q6: Explain why conv-based methods have lower (A_sub) even though they are sensitive to texture information, and sub protocol also uses texture information to composite images. 

- Good question. We are very sory that our explaination in line 268-271 did not satisfy you. We now try to discuss more about this.
- We said ``conv-based models are sensitive to texture information'', thus, they tends to use texture features for prediction. Further, when texture features are absent from the target concept (in sub test, we use objects with different texture features for evaluation, e.g., train using red, black, and white shirts but test the green shirt in sub), models are confused to recognize the target concept.
    - For example, a model recognizes the shirt concept by its color “red or black or white”. When a “green shirt” comes, this model does not recognize it is also a shirt. Thus, it results in poor test acc on sub test.
- One the contrary, vit-based models tend to use shape features for prediction and can correctly recognize “green shirt” as shirt concept since it has the same shape as other shirts.
- On the other hand, we guaranteed the Solvability that the evaluated attribute are seen in other concepts. The pool A_sub results indicate its pool compositionality on attirbute-level.

Q7: Experimental results on ``Sample efficiency for learning compositionality'': why S(sys) are positive when few samples (between 0 to 100 samples) are present?

- Good question. Here I should point out that the samples refer to training samples in the continual training tasks (line 306-307).
- Note that S(sys)=(A_sys-A_non)/A_non. Here non-novel (non) testing tasks contain the same number of labels as training tasks, but the K labels are randomly chosen from the training label pool. That is, it is a small probability that a non-novel testing task is just one of the training task (of course, the number of training samples for each label is different from the training task).
    - Thus, when the model is not well-trained (which is the case when the number of training samples for each continual task is very few (less than 100)), A_non does not necessarily better than A_sys.

Q8: experimental results on ``Varying number of continual training tasks'': why the small-way task needs a smaller number of compositional features for distinguishing classes but the accuracy drops when decreasing number of classes in the task. 

- Good question. First we need to clarify that we train the feature extractor at continual training phase.  For a specific continual task, the model will learn crucial compositional features but miss other compositional features which are not needed for this task but may be crucial for the future continual tasks.
    - Taking the example in Appendix E.5 line 613-616, one can distinguish horse with person by their different shapes. But this is not enough for the case of horse and zebra (i.e., limited compositionality). However, for the tri-classification task of distinguishing among horse, zebra, and person, one can learn both shape and texture features (i.e., relatively better compositionality). The learned texture features can be used in future tasks like distinguishing tigers with other animals.
- Thus, the model may not obtain necessary features for these compositional testing tasks during the continual training phase. As a result, the performance of evaluating compsoitionality (i.e., Hn) drops.

Q9: Purpose to use “concept factorization”. 

- Good question. Sorry for not presenting our motivation to use ``concept factorization''. We will update the motivation in a revision.
- Specifically, we need to mathmatically describe the data generation process from the pespective of sampling distribution. It can clearly shows the difference between our proposed compositional testing tasks.

Thank you again for your comments.

# To Reviewer 6pHG

We sincerely appreciate your constructive comments on this paper. We detail our response below point by point. Please kindly let us know if our response addresses the issues you raised in this paper.

Q1: Grid-like images for multi-task classification and not naturalistic

- Thank you very much on your following designs about other methods to construct benchmarks, which really inspire me and we will carefully consider them.
- Our CGQA is like you said in a grid-like manner. We claim that this is because it provides human interpretable concept visualization and easy to analyse and diagnose the model’s compositionality.
- Also, we provide COBJ, which is not constructed in a grid-like way but a whole real image. Please refer to Appendix C.2 about the construction and image examples.
- Our work is not multi-task classification. As pointed out in Sec 3 Remark 3.2, line 106-111, we do not provide concept labels to the model.
    - As a result, model has to learn the hidden concepts from the image itself.
    - We also discuss about the difficult of learning compositionality under our setting in Appendix A.

Q2: About augmentation

- CGQA is limited to 2x2 grids. Thus, we also tried 3x3 grids called CPIN, constructed from PartImageNet. The details and results were presented in Appendix C.1.7 and E.9, respectively. Since it provides similar conclusions as CGQA, so we only present CGQA in the main paper.
- We summary the training details about augmentation technique we used in Appendix D line 425-434. We use the standard augmentation techniques. And your advices in the Point 4 really inspires me.

Q3: About construction process

- Thank you for your understanding of page limit.
- We presented a short description on Sec 5 line 193-198 about the source dataset we used.

Q4: Missing literature review about augmented-memory-based continual learning

- Thank you very much for providing me these papers. I will improve my paper’s related works part. We will submit our revision as soon as possible.

1和2都没发表，3是pretrained backbone （imagenet）可以跑个结果给他看，但是是做stream leaning的，就是每个task只跑一个epoch，我们的setting是每个tasks最多有100个epoch。setting不一样也没有比的必要。

However, comparing with such pretrained methods are relatively unfair since they may have seen some concepts or concept combinations before.

Thank you again for your comments.

