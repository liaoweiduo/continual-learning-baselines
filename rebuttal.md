# To Reviewer xdij

We sincerely appreciate your constructive comments on this paper. We detail our response below point by point. Please kindly let us know if our response addresses the issues you raised in this paper. 

### Q1: Differences between CFST and multi-label classification tasks
>
> We would like to highlight the key difference between our CFST and the multi-label recognition task, which we have already illustrated in Figure 1 and Remark 3.2 [Lines 106-111] in the main text.
> 
> - In CFST **only the image-level label is available and no concept label is provided**, which is much more challenging than multi-label classification where labels of all concepts in an image are accessible. 
> 
> - Thus, CFST requires **inference of the underlying compositional concepts behind an image**, while multi-label classification does not encourage compositional learning as we have detailed in Appendix A.
>   
>   - Capturing those discriminative features only for classification (e.g., the shape that differentiates "horse" from "human") and failing to capture all the compositional features (e.g., the fine-grained features helpful for identification of "horse") likely struggles in a future CFST task (e.g., classifying between "zebra" and "horse").

### Q2: How CGQA and COBJ reflect practical applications
>
> The objective of our benchmark is exactly to evaluate the capability of a model in compositional learning, i.e., **whether it understands the previously learned concepts and generalizes to future tasks**. We underline that CGQA and COBJ meet this objective in practice:
> 
> - First, we would like to humbly clarify that the task **does not** "train the network to classify each grid", as concept labels are not accessible (see the response to Q1). 
> 
> - Thus, a combined image with only a single label in CGQA is **the same with a natural image in practical applications**, except that we aim to construct a dataset like CGQA with **(1)** all the concepts in an image **easy to parse and interpret**, and **(2)** **a comparably small number of concepts**. CGQA constructed in such a manner serves as a **less challenging benchmark** to evaluate various methods.
> 
> - The **more challenging benchmark of COBJ** contains images that also have **single labels without concept/object labels**; compared to CGQA, the concepts within each image are more complicated. 
>   
>   - We have detailed the construction process of COBJ in Appendix C;
>   
>   - We have provided image examples in Appendix Figure 5.
​
### Q3: Freezing modules limits the adaptability of the methods to new tasks
> 
> - We follow the **original implementation of modularity-based methods** which **freeze old modules to maximally prevent catastrophic forgetting** and learn new modules sub-linearly to solve new knowledge. 
> 
> - Freezing modules is also crucial to **guarantee fair comparison**, so that for each task the number of trainable parameters remains the same and comparable to other baselines. In fact, we had previously jointly trained all modules of RPSnet, resulting in the performance of $H_n=67.99$ on CGQA. Though this is higher than that by freezing old modules ($H_n=59.94$ in our paper), the improvement by greater model capacity is not fair.

### Q4: No discussion about the negative societal impact
>
> - Thank you very much to point out the limitations. Our benchmarks do not contain any sensitive data, such as medical data, which makes them ethically sound for use in research.
> - We will update the discussion in our new revision as soon as possible.

Thank you again for your comments.












# To Reviewer RpQ2

We sincerely appreciate your constructive comments on this paper. We detail our response below point by point. Please kindly let us know if our response addresses the issues you raised in this paper.

### Q1: Model size comparison for RPSnet to other CL methods. 
>
> This is a very good concern. We would like to highlight that we follow the **original implementation of modularity-based methods** which freeze old modules to maximally prevent catastrophic forgetting and **learn new modules sub-linearly** to solve new knowledge. 
>
> - RPSnet (also MNTDP) exactly has much **more model parameters** than other methods when the current task id $t>0$. 
>
> - However, in order to **guarantee fair comparison**, RPSnet only allows to train one module each layer for one task. Overall, **the number of trainable parameters remains the same and comparable to other baselines**. 

### Q2: Limitation on selected concepts. 
>
> This is a very good comment and advice. The reason we chose these ``visual'' concepts is as follows:
>
> - We wanted to achieve **balance** and **flexible** combinations of concepts. That is, the number of instances for different combinations of concepts should be similar (no long-tailed combinations) and we can combine any pair of concepts. 
> 
> - However, some concepts, like motion and human mood, can only be combined with the human concept (maybe it is better to explain as ``attribute'' in this work). This limits the flexibility and limits the number of re-combinations in the systematicity test since it is difficult to find various kinds of moods on other animals (maybe cats can :p ).
> 
> - Our future work is to further study concepts hierarchically to introduce different generalization challenges. We can contain more ``abstract'' (not visual) concepts, e.g., by specifically combining facial or gesture benchmarks.

Thank you again for your comments.





















# To Reviewer 9Qe4

We sincerely appreciate your constructive comments on this paper. We detail our response below point by point. Please kindly let us know if our response addresses the issues you raised in this paper.

### Q1: Move related work from appendix to main paper
>
> - This is a very good advice. Unfortunately, due to the page limit, we did not include them in our main text. However, we clearly understand that they are important for the reader so we will include them in the revision.

### Q2: Too strict requirement of frozen feature extractor
> 
> First I would like to highlight that our benchmarks **do not impose any restriction** on the methods. Maybe our claim in the main paper misleads you, and we will correct the expression in the revision.
> 
> - During **CL training**, the methods’ feature extractor does not have to be frozen (which is also the case in our experiments). 
> 
> - We now explain why we chose to use **few-shot tasks** and **frozen feature extractors** during **evaluation**. 
> 
>   - At the specific checkpoint (after finishing all continual training tasks), we use our evaluation tasks to evaluate the model’s compositionality. In such a condition, if the number of support samples in a evaluation task is large, the feature extractor may learn from them, and thus, we can not actually judge whether the good performance comes from the original feature extractor (learned from old tasks).
> 
>   - We also have tried to not freeze the feature extractor and the results are presented in Appendix E.7. For your convenience, we show our observation: all methods showed a performance drop if not freezing the feature extractor, especially for ER. It was clearly an overfitting issue and this bad effect was method-dependent. So in order to eliminate this effect when comparing the accuracies among methods, we froze the feature extractor.
> 
> - Note that ``frozen feature extractor'' is not a strict requirement, thus, methods that don't explicitly separate feature learning from classifier learning can also use our evaluation protocol.

### Q3: ``Never-ending'' CL sense
> 
> This is a very good comment. Our reported results were indeed static (only evaluating after finishing all continual training tasks), but the setting is clearly a never-ending CL.
> 
> - We claim that our diagnosing evaluation can be used at any checkpoint (not to be restricted at the end of all continual training tasks). We report the per-task Hn on 10-way COBJ as follows:
> 
>    | Methods | Hn_1 | Hn_2 | Hn_3 |
    | -----------  | ---      | ---      | ---     |
    | Finetune  | 37.57 | 38.32 | 37.82 |
    | ER        | 30.27 | 33.33 | 36.99 |
    | GEM       | 38.05 | 37.60 | 37.36 |
    | LwF       | 38.13 | 43.34 | 45.19 |
    | EWC       | 37.34 | 36.86 | 38.68 |
    | Finetune* | 37.41 | 37.07 | 40.93 | 
    | ER*       | 31.11 | 37.30 | 38.66 |
    | GEM*      | 37.45 | 35.60 | 40.93 |
    | LwF*      | 37.84 | 44.01 | 44.75 |
    | EWC*      | 30.76 | 34.18 | 36.20 |
>    
>    - where Hn_1 denotes Hn after finishing the first continual task. We deepcopy the feature extractor and train a new classifier for each compositional testing task.
>
>    - A rough observation: methods tend to improve compositionality when seeing more labels (combinations of concepts). 

### Q4: Disentangled concepts are not real
>
> Thank you very much to point out the limitation. We apologize for not discussing more about this due to the page limit. We now explain why we choose those ''visual'' concepts to construct our benchmarks. 
> 
> - Note that, we want to achieve **balance** and **flexible** combinations of concepts. That is, the number of instances for different combinations of concepts can be similar (no long-tailed combinations) and we can combine any pair of concepts together. 
> 
> - However, some concepts like motion and human mood can only be combined with the human concept (maybe it is better to explain as ''attribute'' in this work). This limits the flexibility and limits the number of re-combinations in the systematicity test since it is difficult to find various kinds of moods on other animals (maybe cats can :p ).

### Q5: Relation to prior work
> 
> Thank you very much to point out our weakness and we will address in our revision. 
>
> - We have mentioned in Sec 2 (Related Works) that some vision benchmarks evaluate compositionality (including works in the CL field) and we have pointed out that some benchmarks are toy and evaluate only systematicity.
> 
> - As for the CL methods, unfortunately, due to the page limit, we apologize for not including a detailed discussion in our main paper. However, we clearly understand that they are important and will include them in the revision. 
>
> - Additionally, as for the relationship with CZSL, we have illustrated the difference in Figure 1 and Remark 3.2.
>
> - As for the relationship between our evaluation protocol with forgetting, we have investigated empirically by a case study in Sec 6 [lines 276-304] and also have provided detailed experiments in Appendix E.4. 

### Q6: Question about non-novel testing accuracy vs training accuracy in Sec 4
> 
> - This is a good question. The answer is yes. Although one non-novel testing task contains the same number of labels as one training task, the K labels are **randomly chosen from the training label pool** and there is no intersection between their sample instances. That is, it is a small probability that the K labels of a non-novel testing task is just the same as that in one training task (of course, the number of training samples for each label is relatively smaller than that in the training tasks). 

### Q7: Put more description of the construction process rather than the motivation in Sec 5
>
> - This is a very good advice. We apologize for not including a detailed construction process in our main paper, due to the page limit. However, we clearly understand that this is very important for the readers to understand our work, thus, we will compress the motivation and put the construction process to the main paper in the revision.

### Q8: Question about experimental results
>
> 1. Multitask is not the best on Hn in COBJ (RPSnet and ER* are the best)
>
>   - Firstly, we would like to highlight that Hn evaluates the **compositionality of the feature extractor**.
>
>   - This observation is quite interesting, that, Multitask may not necessarily be the upper bound in terms of compositionality. In CGQA, compositionality is easier to learn since we visually split the concepts which the models are expected to learn. Multitask shows a great superiority on Hn in CGQA, which is also consistent with the CAM visualization results in Appendix E.3 and Fig7. Multitask could recognize more concepts than Finetune.
>
>   - However, in the real-world case (COBJ), concepts are not as visually separable as that in CGQA. Our CAM visualization results in Appendix Fig8 showed that Multitask was better than Finetune but the gap was not so large (the number of recognized concepts by Multitask was larger but similar to that by Finetune). That is, Multitask might not necessarily beat CL methods in terms of compositionality.
>
>   - Hope my analysis solves your question.
>
> 2. MNTDP* is not the top performer on Hn on COBJ, only on Acon.
>
>   - Good question. As we highlighted in the above question, a model with better Acon does not necessarily have better Hn (compositionality).
>
>   - This result indicated that MNTDP* showed no superiority in compositionality. The high average test accuracy was due to the zero forgetting of old tasks since it froze all learned modules for old tasks.
>
> - The above two observations also indicate the shortcomings of average test accuracy Acon. **Our evaluation metric Hn provides more insights**.

### Q9: Grammar errors and typos
>
> - Sorry for my grammar errors and typos that lead to the misunderstanding of some parts.
>
> - We will carefully check those typos in our revision.

Thank you again for your comments.














# To Reviewer fvLh

We sincerely appreciate your constructive comments on this paper. We detail our response below point by point. Please kindly let us know if our response addresses the issues you raised in this paper.

### Q1: Contradiction between experimental results and the claim: "compositionality addresses the stability-plasticity dilemma" 
>
> This is a very good question. First, we would like to highlight that we are discussing the stability and plasticity of the **feature extractor** of a continual learner. In this case, **average test accuracy is not a suitable metric since it takes account of both the feature extractor and the classifier**. 
> - We analyse the forgetting phenomenon of the average test accuracy as follows: 
>   1. The **feature extractor** forgets the crucial features for old tasks when learning the new task. 
>   2. In a multi-classifier implementation: the **classifiers** of old tasks can not update (no old sample); in a single-classifier implementation: the prediciton will bias to classes in the current task (non-i.i.d). Thus, the coupling between the feature extractor and the corresponding classifiers is broken. 
> - In our experiment, we visualized the CAM of the feature extractor in Appendix E.3. On CGQA, the learned feature extractor of Finetune was compositional, thus, the feature extractor had good stability. However, Acon was very bad, showing that when combined with the classifiers, the stability was poor. Thus, we claim that using Acon to evaluate the feature extractor is faulty. 
> - While our Hn evaluates the compositionality of a feature extractor (how well can this feature extractor extract compositional features). We eliminate the effect of the classifier.
> - By the way, our evaluation method is flexible and can also be used on algorithms that don't explicitly separate feature learning from classifier learning. We just provide few-shot testing tasks and algorithms can just deepcopy and evaluate their models with their own methods.

### Q2: Motivations on the evaluations of three compositinoal capabilities
> 
> - First, we would like to humbly clarify that compositionality is a very important ability for a continual learner (see the response to Q1). And most of the current works (see Sec 2 related works) in vision **only consider systematicity (novel re-combination)** as compositionality. We extend to productivity and substitutivity (other two very interesting aspects of compositionality which are widely studied in the NLP field) to provide more insights.
> - We now discuss more about the motivations for productivity and substitutivity:
>   - **Productivity**: If a feature extractor has good compositionality, the extracted features exactly represent each compositional component. Then it should be easy to generalize to complex images with more seen concepts. For example in our main paper lines 150-151, after gathering knowledge of concepts Door, Shirt, Grass, Table, Hat, Leaves from the task (distinguishing {Door, Shirt}, {Grass, Table}, and {Hat, Leaves}), the model should easily understand {Door, Leaves, Shirt, Table}, although it does not seen any instance of this label before. Our productivity test tasks only contain images with more concepts than the continual training tasks (e.g., images in continual training tasks may only have 2 concepts while images in the productivity test tasks have 3 or 4 concepts). 

    [需要删除：Compositional feature extractors trained with simple combinations of concepts can easily generalize to complex images (more visual concepts). For example in our main paper line 150-151, an un-compositional feature extractor may learn coupled features between Grass and Table concepts when recognizing {Grass, Table}. Then, when seeing {Door, Leaves, Shirt, Table} (one image with the Table concept but no Grass concept), it does not have high activating values on these features. On the other hand, a compositional feature extractor learns decoupled features for Grass and for Table concepts. Thus, it can have higher activating values on Table features when seeing {Door, Leaves, Shirt, Table}. The productivity test is to evaluate this performance.]
>
>   - **Substitutivity**: In order to achieve **balance** and **flexible** combinations of concepts (the number of instances for different combinations of concepts can be similar (no long-tailed combinations) and we can combine any pair of concepts), our selected concepts are all visual and disentangled. However, some concepts (e.g., white color) are more likely to be the ''attribute'' of other concepts (e.g., shirt). These attribute-like concepts are not so flexible that they sometimes accompany by some concrete concepts. To compensate for the evaluation of these attribute-like concepts, we design the substitutivity test. 

### Q3: The proposed benchmarks are not truly in continual learning setting and knowledge leaking on continual training tasks
>
> We are very sorry that our description makes you think we did not conduct a truly continual learning setting. We would like to humbly clarify that **our setting is also a standard CL setting**, which we have illustrated in Figure 1 and Remark 3.2 [Lines 106-111] in the main text.
> 
> - In CFST **only the image-level label is available and no concept label is provided**, thus, CFST requires **inference of the underlying compositional concepts behind an image**. 
>   
> - We compare our benchmarks with a standard CL setting: Split-CIFAR100. Our CGQA has 100 different labels and if the number of tasks is 10, each task will have 10 different labels. These labels are actually the existence of the concepts, thus, different labels may contain overlapped concepts (e.g., there can be two labels: {Door, Shirt} and {Grass, Shirt}). As pointed out in Sec 3 Remark 3.2, lines 106-111, and Figure 1, these concepts are potentially hidden. For example, we can assign label 0 to {Door, Shirt} and label 1 to {Grass, Shirt}. They are totally different labels and we do not tell the models that label 0 and label 1 all have the Shirt concept. 
>   - This is just the same as Split-CIFAR100 with 100 labels and these labels are evenly distributed in 10 tasks. 
>   - There are two labels (i.e., pine_tree, oak_tree) in CIFAR100. They can be assigned to different tasks in the Split-CIFAR100 setting. These two labels also have overlapped concepts (e.g., leaves, trunk), and the concepts are potentially hidden. 
> - By the way, we also evaluate unseen concepts on noc (non-compositional testing). 

### Q4: Explain why Principle 2 (few-shot learning) and Principle 3 (frozen feature extractor) supports evaluating model's compositionality.
>
> In order to evaluate model's compositionality, we exactly try to question the model that **whether it understands the previously learned concepts and generalizes to future tasks (unseen re-combination of concepts)**, which we have already list the reasons in the main paper line 120-126. 
>
> - At the specific checkpoint (e.g., after finishing continual training tasks), we adopt additional evaluation tasks for this purpose. In such condition, if the number of support samples in the evaluation task is large, the feature extractor may learn from them and thus we can not actually judge **whether the good performance comes from the original feature extractor (learned from old tasks)**. Thus, we recommend few-shot evaluation tasks and frozen feature extractors. 
>
> - We should highlight that **the principles are not strict** and we also did experiments on not frozen feature extractors in Appendix E.7. For your convenience, we show our observation: all methods show a performance drop if not freezing the feature extractor, especially for ER. It is clearly an overfitting issue and the bad effect is method-dependent. So in order to eliminate this effect when comparing the methods and let the accuracy correctly represents the compositionality, we freeze the feature extractor.
>
> - On the other hand, **few-shot tasks can help evaluate the plasticity** since models can have good accuracy on these tasks only if they can fast adapt previous knowledge to the new one. In this case, the compositionality is crucial to fill the systematic gap between the continual training and the few-shot testing tasks. 

### Q5: Add literature review of more recent works and baseline experiments
> 
> - Thank you very much for providing recent works. I will put them into our related works. We will submit our revision as soon as possible.
> - We also run quick experiments on codaPrompt, dualPrompt, l2p++, deep l2p++, and the corresponding finetune method with pretrained backbone (**FT_Classifier**: freeze feature extractor and finetune classifier; **l2p++**: use prefix-tuning instead of prompt-tuning; **deep l2p++**: add prefix-tuning at all layers). The results are as follows:
>   
>   | CGQA          | Acon| sys | pro | sub | Hn | non | noc | Hr | Ha |
>    |---------------| --- | --- | --- | --- | --- | --- | --- | --- | --- |
>    | dual-prompt   | 85.52 ± 1.47 | 65.98 ± 1.68 | 69.32 ± 1.61 | 76.72 ± 1.56 | 70.40 | 69.26 ± 1.63 | 84.56 ± 1.22 | 76.15 | 72.59 |
>    | coda-prompt   | 77.43 ± 1.91 | 52.24 ± 1.46 | 53.96 ± 1.77 | 62.14 ± 1.60 | 55.80 | 54.50 ± 1.56 | 74.38 ± 1.76 | 62.91 | 58.44 |
>    | l2p++         | 83.02 ± 1.66 | 61.46 ± 1.54 | 63.02 ± 1.61 | 71.28 ± 1.60 | 64.98 | 64.72 ± 1.75 | 81.70 ± 1.42 | 72.23 | 67.70 |
>    | deep l2p++    | 77.84 ± 1.91 | 52.22 ± 1.60 | 54.52 ± 2.00 | 62.70 ± 1.71 | 56.14 | 54.24 ± 1.58 | 74.24 ± 1.50 | 62.68 | 58.58 |
>    | FT_Classifier | 78.13 ± 1.85 | 52.34 ± 1.51 | 54.04 ± 1.28 | 61.04 ± 1.47 | 55.56 | 54.14 ± 1.69 | 74.40 ± 1.82 | 62.67 | 58.20 |
>    
>   | COBJ          | continual | sys | pro | Hn | non | noc | Hr | Ha | 
>    |---------------| --- | --- | --- | --- | --- | --- | --- | --- |
>    | dual-prompt   | 90.00 ± 3.63 | 62.48 ± 1.97 | 48.94 ± 2.62 | 54.89 | 51.80 ± 2.36 | 84.04 ± 1.46 | 64.09 | 59.13 |
>    | coda-prompt   | 89.20 ± 3.89 | 60.46 ± 2.10 | 46.76 ± 2.68 | 52.73 | 49.84 ± 2.24 | 83.28 ± 1.61 | 62.36 | 57.14 |
>    | l2p++         | 89.37 ± 3.46 | 61.52 ± 2.11 | 47.50 ± 2.59 | 53.61 | 50.60 ± 2.49 | 83.48 ± 1.42 | 63.01 | 57.93 |
>    | deep l2p++    | 89.90 ± 3.35 | 60.68 ± 2.04 | 46.54 ± 2.55 | 52.68 | 49.22 ± 2.33 | 82.78 ± 1.38 | 61.73 | 56.85 |
>    | FT_Classifier | 89.07 ± 3.98 | 60.62 ± 1.97 | 46.78 ± 2.69 | 52.81 | 48.96 ± 2.25 | 83.44 ± 1.36 | 61.71 | 56.91 |
>
>  - These prompt-based methods utilize a pretrained backbone and learn to extract knowledge from the backbone by prompting. It is clear that these pretrained methods has better Acon and better Hr than the methods we reported in the main text. However, Hn does not outperformed and even is worse than the methods we reported in the main text, which indicates that **the good test accuracy comes from the strong pretrained backbone but the compositionality is not better**. 
> 
>  - Further, the good ''noc'' shows that they have potentially seen these concepts before (those from-scratch learning methods reported in our paper generally have poor ''noc'' performance). 
>
>    - Thus, we claim that **the pretrained backbone may potentially see the labels for testing before** which is unfair to those from-scratch learning methods (baselines I used in our experiments).

### Q6: Provide more justification on the claim "forgetting is not as suffered as that in the class-IL setting on CGQA (Line 244-245)".
> 
> - Sorry, my wrong grammar leads to the misunderstanding. We would like to correct our claim: **''This is because forgetting on CGQA is not as suffered as that on COBJ, especially in the task-IL setting.''** We will update this in our revision.
> 
> - To justify this, we show the test accuracies for Finetune **just after finishing each continual training task** as follows:
>   - task-IL 10-way CGQA tasks
>        
>     | evaluate on task | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
>     |---------------| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
>     | finish task 1 | 58.4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
>     | 2             |55.2 | 66.7 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
>     | 3             | 55.1 | 65.4 | 74.4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
>     | 4             | 54.2 | 63.3 | 61.3 | 83.8 | 0 | 0 | 0 | 0 | 0 | 0 |
>     | 5             | 54.7 | 56.4 | 59.4 | 71.9 | 75.4 | 0 | 0 | 0 | 0 | 0 |
>     | 6             | 48.9 | 57.7 | 66.0 | 71.1 | 72.2 | 77.1 | 0 | 0 | 0 | 0 |
>     | 7             | 54.0 | 58.7 | 63.8 | 75.9 | 66.6 | 72.4 | 75.3 | 0 | 0 | 0 |
>     | 8             | 47.2 | 58.4 | 54.1 | 74.3 | 64.9 | 71.1 | 71.0 | 74.8 | 0 | 0 |
>     | 9             | 61.6 | 70.0 | 68.2 | 82.9 | 76.4 | 76.4 | 75.5 | 69.9 | 84.8 | 0 |
>     | 10            | 53.4 | 69.1 | 74.3 | 78.3 | 75.5 | 75.5 | 70.1 | 67.1 | 82.5 | 82.2 |
>
>   - task-IL 10-way COBJ tasks
>
>     | evaluate on task | 1 | 2 | 3 |
>     |---------------| --- | --- | --- |
>     | finish task 1 | 54.1 | 0 | 0 |
>     | 2             |28.1 | 55.1 | 0 | 
>     | 3             | 35.5 | 29.5 | 53.3 | 
>
> - On CGQA, forgetting is relatively smaller than COBJ even for the naive Finetune method. It is intuitive since COBJ is a real-world benchmark and CGQA is a synthesized benchmark.
> - The key point of MNTDP* to address catastrophic forgetting is to freeze old modules, thus, **it can achieve no forgetting**. However, **in our CGQA case, forgetting is not as suffered as the real-world benchmark (e.g., COBJ)**. Thus, the advantages of MNTDP* are not obvious on CGQA. However, **in COBJ, MNTDP can largely eliminate forgetting**, thus, outperforms the others.

### Q7: Explain why conv-based methods have lower (A_sub) even though they are sensitive to texture information, and sub protocol also uses texture information to composite images. 
>
> - Good question. We are very sorry that our explanation in line 268-271 did not satisfy you. We now try to discuss more about this.
> - We said ``conv-based models are sensitive to texture information'', thus, they tend to use texture features for prediction. Further, when texture features are absent from the target concept (in the sub test, we use objects with different texture features for evaluation, e.g., train using red, black, and white shirts but test the green shirt in sub), models are confused to recognize the target concept.
>   - For example, a model recognizes the shirt concept by its color “red or black or white”. When a “green shirt” comes, this model does not recognize that it is also a shirt. Thus, it results in poor test acc on the sub test.
> - On the contrary, vit-based models tend to use shape features for prediction and can correctly recognize “green shirt” as the shirt concept since it has the same shape as other shirts.
> - Note that, we guaranteed the Solvability that the evaluated attributes are seen in other concepts. The pool A_sub results indicate its pool compositionality on the attirbute level.

### Q8: Experimental results on ``Sample efficiency for learning compositionality'': why S(sys) are positive when few samples (between 0 to 100 samples) are present?
>
> - Good question. I should point out that the samples here refer to training samples in the continual training tasks (line 306-307).
> - Note that $S(sys)=(A_{sys}-A_{non})/A_{non}$. Here non-novel (non) testing tasks contain the same number of labels as training tasks, but the K labels are randomly chosen from the training label pool. That is, it is a small probability that a non-novel testing task is just one of the training tasks (of course, the number of training samples for each label is relatively smaller than the training tasks).
>   - Thus, when the model is not well-trained (which is the case when the number of training samples for each continual task is very few (less than 100)), A_non does not necessarily better than $A_{sys}$.

### Q9: experimental results on ``Varying number of continual training tasks'': why the small-way task needs a smaller number of compositional features for distinguishing classes but the accuracy drops when decreasing number of classes in the task. 
>
> - Good question. First, we need to clarify that we train the feature extractor in the continual training phase. For a specific continual task, the model will learn crucial compositional features but miss other compositional features which are not needed for this task but may be crucial for future tasks.
>   - Taking the example in Appendix E.5 lines 613-616, one can distinguish a horse from a person by their different shapes. But this is not enough for the case of horse and zebra (i.e., limited compositionality). However, for the tri-classification task of distinguishing between horse, zebra, and person, one can learn both shape and texture features (i.e., relatively better compositionality). The learned texture features can be used in future tasks like distinguishing tigers from other animals.
> - Thus, the model may not obtain the necessary features for these compositional testing tasks during the continual training phase. As a result, the performance of evaluating compositionality (i.e., Hn) drops.

### Q10: Purpose to use “concept factorization”. 
>
> - Good question. Sorry for not presenting our motivation to use ``concept factorization''. We will update the motivation in a revision.
> - Specifically, we need to mathematically describe the data generation process from the perspective of sampling distribution. It can clearly show the difference between our proposed compositional testing tasks.

Thank you again for your comments.
















# To Reviewer 6pHG

We sincerely appreciate your constructive comments on this paper. We detail our response below point by point. Please kindly let us know if our response addresses the issues you raised in this paper.

### Q1: The datasets are limited to grid-like images for multi-task classification and images are not naturalistic
>
> We would like to highlight the key difference between our CFST and the multi-task classification / multi-label recognition task, which we have already illustrated in Figure 1 and Remark 3.2 [Lines 106-111] in the main text.
> 
> - In CFST **only the image-level label is available and no concept label is provided**, which is much more challenging than multi-label classification where labels of all concepts in an image are accessible. 
> 
> - Thus, CFST requires **inference of the underlying compositional concepts behind an image**, while multi-label classification does not encourage compositional learning as we have detailed in Appendix A.
>   
>   - Capturing those discriminative features only for classification (e.g., the shape that differentiates "horse" from "human") and failing to capture all the compositional features (e.g., the fine-grained features helpful for identification of "horse") likely struggles in a future CFST task (e.g., classifying between "zebra" and "horse").
>
> The objective of our benchmark is exactly to evaluate the capability of a model in compositional learning, i.e., **whether it understands the previously learned concepts and generalizes to future tasks**. We underline that CGQA and COBJ meet this objective in practice:
> 
> - A combined image with only a single label in CGQA is **the same with a natural image in practical applications**, except that we aim to construct a dataset like CGQA with **(1)** all the concepts in an image **easy to parse and interpret**, and **(2)** **a comparably small number of concepts**. CGQA constructed in such a manner serves as a **less challenging benchmark** to evaluate various methods.
> 
> - The **more challenging benchmark of COBJ** contains images that also have **single labels without concept/object labels**; compared to CGQA, the concepts within each image are more complicated and realistic. 
>   
>   - We have detailed the construction process of COBJ in Appendix C that we crop the region with target concepts on one image, thus, images in COBJ are not grid-like;
>   
>   - We have provided image examples in Appendix Figure 5. 
​>
> By the way, thank you very much for your following designs about other methods to construct benchmarks, which really inspire me and we will carefully consider them. 

### Q2: About augmentations during training
> 
> - In our experiments, we perform the random horizontal flipping for methods with ResNet-18 backbones and additionally perform the Rand-Augment and random erasing for methods with ViT backbones, which we have already summarized in Appendix D line 425-434. 
>
> - We also try to perform augmentation of randomly permutate grid locations. The results are shown below, and for comparison, we also list the results we reported in the main text: 
> 
> | Aug       |   continual | sys           | pro           | sub           |    Hn | non           | noc           |    Hr |    Ha |
  |:----------|------------:|:--------------|:--------------|:--------------|------:|:--------------|:--------------|------:|------:|
  | Finetune  |        8.42 | 63.96 +- 1.73 | 67.22 +- 1.75 | 59.36 +- 1.85 | 65.55 | 67.58 +- 2.13 | 39.34 +- 2.04 | 49.73 | 56.55 |
  | ER        |       18.05 | 73.86 +- 1.63 | 75.18 +- 1.62 | 63.86 +- 1.81 | 74.51 | 77.72 +- 1.54 | 40.92 +- 1.85 | 53.61 | 62.36 |
  | Finetune* |       72.13 | 67.66 +- 2.07 | 71.28 +- 1.62 | 62.06 +- 1.74 | 69.42 | 72.32 +- 1.78 | 40.22 +- 1.84 | 51.69 | 59.26 |
  | ER*       |       77.88 | 72.28 +- 1.66 | 75.42 +- 1.47 | 67.82 +- 1.52 | 73.82 | 77.50 +- 1.72 | 43.62 +- 1.81 | 55.82 | 63.57 |

> | Paper     |   continual | sys           | pro           | sub           |    Hn | non           | noc           |    Hr |    Ha |
  |:----------|------------:|:--------------|:--------------|:--------------|------:|:--------------|:--------------|------:|------:|
  | Finetune  |        8.38 | 64.73 +- 0.78 | 65.43 +- 0.73 | 61.26 +- 0.67 | 63.75 | 68.54 +- 0.80 | 40.32 +- 0.72 | 50.77 | 57.84 |
  | ER        |       19.78 | 71.38 +- 0.75 | 70.11 +- 0.64 | 64.32 +- 0.69 | 68.46 | 77.27 +- 0.67 | 40.98 +- 0.72 | 53.56 | 61.60 |
  | Finetune* |       72.46 | 70.32 +- 0.73 | 72.62 +- 0.63 | 66.33 +- 0.69 | 69.66 | 75.32 +- 0.70 | 43.26 +- 0.73 | 54.95 | 62.92 |
  | ER*       |       76.05 | 71.37 +- 0.70 | 72.67 +- 0.69 | 66.80 +- 0.63 | 70.19 | 76.28 +- 0.66 | 45.61 +- 0.77 | 57.09 | 64.29 |
>
>   - The results are quite similar with and without changing grid locations. 
>
> - As for expanding grid sizes and introducing some distractor image patches, we do not recommend to do that. The reasons are as follows:
> 
>   - The provided image-level labels are the existence of all concepts in the images, thus, introducing other image patches may potentially change the number of existence concepts. 
> 
>   - While our Productivity test aims to evaluate the generalization capability on more concepts, we need to prevent the model from potentially seeing more concepts in one image during continual training. 
>
> - By the way, we also tried another benchmark with 3x3 grid-like image, called CPIN, constructed from PartImageNet. The details and results have been presented in Appendix C.1.7 and E.9, respectively. Since it provides similar conclusions as CGQA, so we only present CGQA in the main paper.

### Q3: About construction process
> 
> - Thank you very much for your understanding of the page limit. We provide a short description on Sec 5 [lines 193-198] about the source dataset we used.
> 
> - We also clearly understand that the construction processes are important for readers to understand our work. We will put these parts to the main text in our revision. 

### Q4: Missing literature review about augmented-memory-based continual learning
> 
> - Thank you very much for providing me with these papers. I will include them and improve the related works part. We will submit our revision as soon as possible.
>
> - We run REMIND on our CGQA and the results were shown below: [run on our setting, without ImageNet pretrained feature extractor(first 4 layers), waiting for analysis]

> | CGQA    | Acon| sys | pro | sub | Hn | non | noc | Hr | Ha |
  |---------| --- | --- | --- | --- | --- | --- | --- | --- | --- |
  | REMIND  | 8.34 | 10.16 +- 0.71 | 9.90 +- 0.81 | 9.10 +- 0.65 | 9.70 | 9.06 +- 0.70 | 9.40 +- 0.60 | 9.23 | 9.50 |
  - For the time limit, we just ran the same stream learning setting as in the REMIND paper. So it was not fair directly compare it with the results in our paper. 
  - It seems that REMIND did not perform well when evaluating with few-shot tasks since samples were only seen once in the stream learning setting. 

Thank you again for your comments.


