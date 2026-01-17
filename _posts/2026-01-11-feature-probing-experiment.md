---
layout: post
title: "Can a model trained to predict letters learn to represent words?"
date: 2026-01-11
---
<nav>
  <a href="{{ site.baseurl }}/">← Back to all posts</a>
</nav>

<hr>
I recently did a fun weekend experiment to test out the use of sparse autoencoders for feature probing.  Specifically, I train a model to predict individual characters from the TinyStories dataset, and then use feature probing to test if the model has learned to represent individual words from this simple objective.  The purpose of this experiment is to build an intuition on the development of features, and more importantly for me, to start experimenting in the space of emergent language abilities.  That is to say, it is a first and smallest step in the direction of having models learn language as a higher order abstraction of environmental features.  

I am interested in this because I am interested in how we (as humans) can learn distinct concepts (that we can put words to) from raw enviromental data -to the extent that language emerges naturally from this process.  I think this is an interesting metric to pose for models -that is, if a model can produce language as an emergent property of modeling environmental data, operating as a agent in that enviromemnt, and performing social functions with other agents, then it surely has achieved impressive levels of intelligence.  And this leads to the question that truly interests me, which is how we can structure models to do this.  An interesting aside: did you know that 2 human babies can develop their own language to communicate with each other absent any other exposure to language?  Imagine if two artificial models could do this also!

The scope of this experiment does not reach this yet, as I said, this is a first and smallest step in my exploration of this idea.

Going into this experiment, I fully expected to the model to learn to represent words, as doing so seems to be the most natural way to properly predict characters -if you have an internal model of words and their composed characters, predicting next characters is a trivial task. 

#Experimental Design

##Transformer Model
I trained a simple GPT-2 style model to predict character level (byte) tokens from the TinyStories dataset.  I trained the model for 6000 steps, with a batch size of 32, and context length of 512, totalling around 100 000 000 tokens.  The model was trained on a single Tesla T4 GPU for a grand total of a couple dollars worth of compute credits.

Model hyperparameters table below:

| Hyperparameter | Value |
| --- | --- |
| Vocab Size | 256 |
| Context Length | 512 |
| Embedding Dimension | 512 |
| Attention Heads | 8 |
| Layers | 8 |
| FFN Hidden Dimension | 2048 |
| Dropout | 0.1 |
| Normalization | RMSNorm | 
| Positional Embedding | RoPE 
| Optimizer | AdamW |
| Batch Size | 32 |
| Training Steps | 6000 |
| Gradient Accumulation Steps | 4 |
| Learning Rate | 1e-3 |
| Weight Decay | 0.1 |
| Warmup Steps | 300 |
| Learning Rate Scheduler | Cosine |

The model reached a final validation loss of 0.5750 after training, which I figured would be sufficient for the task at hand.  The text generation is thematically relevant to the prompt but kind of incoherent as well, but notable the byte level generation produces real words, spelt correctly and relevant to the context.  
Eg:
>Prompt: In a world where artificial intelligence
>Output: In a world where artificial intelligence and a whale wanted to be saved!
>
>The fish and the whale worked together to build a river. Soon the whale was sailing around the river with soap!

This definitly has that delightful creativity that only small models seem to produce.  

<figure>
  <img src="/_images/char-feature-model-val-loss.png" alt="Loss curve for character prediction model.">
  <figcaption>Loss curves for character prediction model.</figcaption>
</figure>

##Sparse Autoencoder
I then trained the SAE.  I used <a href="https://cdn.openai.com/papers/sparse-autoencoders.pdf">top-k activation selection to enforce sparcity</a>. I used a 32x expansion factor, for a hidden dimension of 16384, and a k value of 32.

SAE hyperparameters table below:

| Hyperparameter | Value |
| --- | --- |
| Expansion Factor | 32 |
| Hidden Dimension | 16384 |
| k | 32 |
| Learning Rate | 1e-4 |
| Batch Size | 4096 |
| Training Steps | 30 000 |
| Warmup Steps | 100 |
| Resample Dead Neurons | True |
| Dead Neuron Threshold | 10000 |

One mistake I made was only collecting 250 000 activations for feature probing, which is probably 100x to few given the number of training steps.  I don't think this really affects my results for the purposes of this experiment, but it is something I will be more careful about next time.
Anyways, the SAE model reached a final reconstruction loss of 0.6197, an aux top-k loss of 0.4488.

<figure>
  <img src="/_images/sae-recon-loss.png" alt="Training stats for SAE model.">
  <figcaption>Graphs of training stats for SAE model. These graphs reflect performance on the training set</figcaption>
</figure>

I then did a basic analysis of the activations, finding:

>Total features: 16384
>Dead features (never activated): 2844
>Rare features (<0.1%): 10311
>Mean activation frequency: 0.0020
>Median activation frequency: 0.0007

I then look to find which features activate very strongly to particular words, and which features activate exvlusively to particular words.  I look at the most common words in my dataset, and find the features that activate for tokens towards the end of the word. This roughly corresponds to the presence of features that can detect the end of a word.  A second exmperiment I am currently running looks at the middle of the word and examines the features produced there.  I will update this post with the results of that experiment when I am finished with it.
<figure>
  <img src="/_images/feature-activation-frequency.png" alt="Distribution of feature activations and their frequency.">
  <figcaption>Distribution of feature activations and their frequencies.</figcaption>
</figure>
#Results
Overall I found 262 words that have unique highly selective features active strongly for them.  My analysis so far isn't super deep, but superficially these results are seem pretty strong.  If anything, I am quite impressed by how many words have dedicated highly selective features.  

I am also currently running a second experiment where I test the effects of injecting these features into the model.  I want to see to what extent that I can control the world level behaviour of the model by injecting these features.  I will update this post with the results of that experiment when I am finished with it.

Below is a list of all the words with highly selective features

| Word | Occurrences | Dedicated_Features | Features_List (activation strength) |
|------|-------------|--------------------|---------------|
| was | 32063 | 27 | F16208(100%), F6144(100%), F12773(100%)...|
| it | 28110 | 26 | F12012(100%), F5256(100%), F9934(99%)...  |
| to | 34514 | 25 | F4634(100%), F7603(100%), F14879(100%)... |
| of | 6736 | 19 | F16318(100%), F11630(100%), F5713(97%)... |
| a | 21977 | 18 | F8611(100%), F6227(100%), F9030(100%)... |
| they | 13743 | 18 | F11398(100%), F13174(100%), F12995(100%)... |
| you | 8162 | 17 | F12040(100%), F15938(100%), F13081(99%)... |
| once | 4373 | 15 | F1366(100%), F4684(100%), F839(99%)... |
| day | 5803 | 14 | F9097(100%), F7205(100%), F411(100%)... |
| and | 46952 | 14 | F1064(100%), F16323(100%), F1645(97%)... |
| little | 9825 | 13 | F10811(99%), F7946(99%), F7837(99%)... |
| but | 9042 | 12 | F13376(100%), F15656(99%), F6409(98%)... |
| she | 18484 | 11 | F12464(100%), F9386(100%), F15999(100%)... |
| said | 6108 | 11 | F10007(100%), F15194(100%), F10381(97%) |
| girl | 5535 | 10 | F3698(100%), F482(100%), F11069(92%)... |
| the | 43122 | 9 | F6785(100%), F7881(97%), F15744(96%)... |
| her | 12147 | 9 | F9707(100%), F13146(98%), F7238(95%)... |
| one | 4479 | 9 | F8079(100%), F13718(100%), F15025(100%)... |
| that | 7581 | 8 | F16012(100%), F671(99%), F12519(96%)... |
| there | 4383 | 8 | F12393(100%), F1566(100%), F12899(99%)... |
| back | 4220 | 8 | F6363(97%), F16116(96%), F2357(92%)... |
| very | 4667 | 7 | F13015(100%), F1015(98%), F15993(97%)... |
| sam | 2550 | 7 | F6875(100%), F16100(100%), F15231(99%)... |
| had | 6120 | 7 | F478(100%), F11821(95%), F2828(93%)... |
| t | 7714 | 7 | F2462(97%), F13242(94%), F5314(83%)... |
| so | 11356 | 6 | F8646(100%), F13411(91%), F7872(86%)... |
| for | 3983 | 6 | F5600(100%), F3952(100%), F1387(92%)... |
| s | 3992 | 6 | F15675(100%), F4455(95%), F12240(94%)... |
| jungle | 1521 | 6 | F3018(100%), F9798(94%), F13165(94%)... |
| dry | 4590 | 6 | F7009(95%), F6732(94%), F5997(90%)... |
| he | 24120 | 5 | F13623(100%), F11294(84%), F13504(81%)... |
| big | 1530 | 5 | F2993(100%), F12873(99%), F15614(92%)... |
| upon | 3360 | 5 | F5369(100%), F7651(95%), F972(91%)... |
| their | 1524 | 5 | F14506(100%), F5197(98%), F6636(96%)... |
| can | 3294 | 5 | F6044(99%), F586(98%), F3958(98%)... |
| license | 2530 | 5 | F8561(99%), F16326(98%), F14412(86%)... |
| mom | 4590 | 5 | F8609(99%), F7240(91%), F15899(86%)... |
| other | 1524 | 5 | F3036(93%), F6538(91%), F8254(86%)... |
| how | 3062 | 4 | F5964(100%), F15077(98%), F6285(87%)... |
| friends | 2024 | 4 | F12602(100%), F15692(89%), F16074(87%)... |
| cry | 510 | 4 | F2315(100%), F2060(83%), F259(77%)... |
| sad | 2121 | 4 | F4534(100%), F5447(100%), F293(85%)... |
| put | 2040 | 4 | F9821(99%), F14648(98%), F12378(86%)... |
| plant | 2540 | 4 | F5520(99%), F10756(98%), F6252(80%)... |
| park | 2036 | 4 | F9344(99%), F11626(99%), F13898(99%)... |
| saw | 1530 | 4 | F1505(99%), F11129(82%), F2765(78%)... |
| were | 7126 | 4 | F2718(99%), F667(93%), F13203(80%)... |
| break | 1016 | 4 | F4389(97%), F7842(85%), F9686(80%)... |
| new | 1531 | 4 | F2659(97%), F12094(96%), F7777(88%)... |
| pretty | 2028 | 4 | F9495(95%), F11519(92%), F12983(86%)... |
| tree | 3054 | 4 | F16136(92%), F12271(89%), F1035(81%)... |
| toys | 1018 | 4 | F12437(92%), F8125(79%), F16241(77%)... |
| after | 508 | 3 | F2525(100%), F14952(95%), F5619(82%) |
| hard | 1527 | 3 | F12064(100%), F6377(97%), F11740(80%) |
| ran | 2520 | 3 | F11022(100%), F13828(81%), F14320(76%) |
| time | 4480 | 3 | F7334(100%), F12277(94%), F8792(84%) |
| him | 2460 | 3 | F477(100%), F7762(79%), F13708(75%) |
| goodbye | 1518 | 3 | F1214(99%), F14499(97%), F9107(87%) |
| alone | 1156 | 3 | F8436(99%), F1016(79%), F526(79%) |
| sport | 1016 | 3 | F279(99%), F673(98%), F1265(96%) |
| have | 4581 | 3 | F10608(99%), F1480(94%), F7728(90%) |
| brown | 508 | 3 | F14119(98%), F3243(90%), F12(78%) |
| this | 2036 | 3 | F6091(98%), F4711(90%), F13460(89%) |
| fence | 3556 | 3 | F5511(98%), F12117(88%), F12782(83%) |
| water | 2033 | 3 | F14191(98%), F7951(91%), F5014(90%) |
| costume | 506 | 3 | F5130(98%), F1673(89%), F5408(88%) |
| driver | 1014 | 3 | F4949(98%), F15863(83%), F13996(76%) |
| what | 4581 | 3 | F4597(98%), F3421(91%), F2657(75%) |
| fight | 1016 | 3 | F14173(98%), F8761(95%), F7359(90%) |
| his | 9606 | 3 | F4834(97%), F13805(88%), F3083(86%) |
| loud | 848 | 3 | F6188(97%), F4532(96%), F5579(82%) |
| statue | 2028 | 3 | F15354(94%), F1839(81%), F8410(79%) |
| name | 1367 | 3 | F6533(94%), F141(91%), F13853(80%) |
| from | 1455 | 3 | F1529(92%), F14848(90%), F3979(84%) |
| fun | 3570 | 3 | F8380(90%), F7347(83%), F1802(79%) |
| party | 1016 | 3 | F3559(89%), F2885(83%), F16094(82%) |
| get | 2041 | 3 | F5178(85%), F1746(77%), F2544(76%) |
| too | 2552 | 3 | F8124(84%), F8644(80%), F16314(78%) |
| would | 3992 | 3 | F15764(83%), F2162(79%), F3071(78%) |
| soon | 2545 | 3 | F9174(82%), F12290(81%), F7751(76%) |
| rabbit | 507 | 3 | F5709(80%), F618(76%), F781(76%) |
| wanted | 1919 | 2 | F8935(100%), F10273(86%) |
| only | 1018 | 2 | F5635(100%), F11799(94%) |
| lots | 509 | 2 | F686(100%), F11933(75%) |
| curious | 1012 | 2 | F10062(100%), F7298(99%) |
| rain | 2546 | 2 | F3056(100%), F14910(83%) |
| order | 1524 | 2 | F6382(100%), F7072(75%) |
| well | 509 | 2 | F8104(100%), F6036(78%) |
| paper | 508 | 2 | F12825(99%), F2256(76%) |
| risk | 509 | 2 | F7060(99%), F12002(99%) |
| like | 3083 | 2 | F14623(99%), F9190(88%) |
| decorations | 1506 | 2 | F952(99%), F3985(85%) |
| long | 1018 | 2 | F7781(99%), F14317(99%) |
| gave | 509 | 2 | F12403(99%), F13090(78%) |
| bats | 509 | 2 | F7756(98%), F11943(95%) |
| ah | 511 | 2 | F9596(98%), F1164(90%) |
| up | 2561 | 2 | F15296(98%), F4128(90%) |
| special | 1518 | 2 | F11695(98%), F10949(77%) |
| bubby | 2032 | 2 | F11831(98%), F4183(78%) |
| went | 3253 | 2 | F5249(97%), F9129(80%) |
| threw | 508 | 2 | F103(96%), F11541(79%) |
| mum | 3060 | 2 | F6077(96%), F5721(85%) |
| down | 1018 | 2 | F10908(96%), F985(86%) |
| then | 2036 | 2 | F5383(96%), F2646(93%) |
| kids | 3916 | 2 | F760(95%), F9810(88%) |
| game | 1018 | 2 | F2620(95%), F2904(80%) |
| away | 2581 | 2 | F8814(94%), F5325(90%) |
| mama | 2545 | 2 | F4757(94%), F2994(77%) |
| famous | 1211 | 2 | F11279(94%), F15912(88%) |
| henry | 2033 | 2 | F1369(93%), F11502(79%) |
| himself | 506 | 2 | F5912(93%), F5538(82%) |
| make | 1018 | 2 | F8374(92%), F7799(80%) |
| soar | 509 | 2 | F1243(92%), F16185(81%) |
| leaves | 507 | 2 | F7062(91%), F16008(82%) |
| valuable | 505 | 2 | F4361(91%), F13605(82%) |
| around | 3943 | 2 | F3449(90%), F11869(76%) |
| much | 1527 | 2 | F1020(90%), F4097(88%) |
| no | 1941 | 2 | F8855(88%), F2509(78%) |
| my | 1022 | 2 | F4585(88%), F3984(78%) |
| dark | 1326 | 2 | F4973(88%), F6054(87%) |
| hurt | 1194 | 2 | F8285(87%), F14571(83%) |
| home | 2545 | 2 | F9534(86%), F3634(85%) |
| adult | 2032 | 2 | F1320(86%), F14228(83%) |
| come | 2036 | 2 | F14157(84%), F14112(79%) |
| because | 1012 | 2 | F5183(84%), F4528(79%) |
| story | 508 | 2 | F13896(83%), F6963(80%) |
| with | 6618 | 2 | F12301(82%), F803(79%) |
| need | 1527 | 2 | F994(81%), F5940(76%) |
| jack | 2902 | 2 | F8776(81%), F15280(80%) |
| at | 3094 | 2 | F862(80%), F8655(78%) |
| sky | 510 | 2 | F6229(78%), F1685(76%) |
| floppy | 3042 | 1 | F12827(100%) |
| happened | 505 | 1 | F11805(100%) |
| as | 2453 | 1 | F2325(100%) |
| loved | 508 | 1 | F4550(100%) |
| stopped | 506 | 1 | F9773(100%) |
| whale | 293 | 1 | F9363(100%) |
| build | 508 | 1 | F6496(100%) |
| came | 1530 | 1 | F15895(100%) |
| hello | 1016 | 1 | F8947(99%) |
| pond | 509 | 1 | F8348(99%) |
| an | 1647 | 1 | F10322(99%) |
| important | 504 | 1 | F5355(99%) |
| them | 3054 | 1 | F7102(99%) |
| decided | 1518 | 1 | F5566(99%) |
| before | 1014 | 1 | F15724(99%) |
| joy | 1020 | 1 | F15924(98%) |
| named | 508 | 1 | F11392(98%) |
| different | 504 | 1 | F5445(98%) |
| metal | 508 | 1 | F4555(98%) |
| world | 1557 | 1 | F7550(98%) |
| sweet | 1524 | 1 | F8903(98%) |
| been | 1527 | 1 | F8886(98%) |
| both | 1018 | 1 | F6908(97%) |
| keen | 509 | 1 | F5048(97%) |
| brought | 506 | 1 | F3040(97%) |
| strange | 506 | 1 | F7509(97%) |
| still | 508 | 1 | F920(97%) |
| sunny | 508 | 1 | F9904(96%) |
| folded | 1014 | 1 | F3017(96%) |
| tool | 790 | 1 | F9254(96%) |
| high | 509 | 1 | F5506(96%) |
| fruit | 508 | 1 | F7342(96%) |
| if | 2044 | 1 | F1791(95%) |
| johnny | 3042 | 1 | F13467(95%) |
| proud | 508 | 1 | F3900(95%) |
| voice | 508 | 1 | F13606(95%) |
| each | 2037 | 1 | F2448(94%) |
| hugging | 506 | 1 | F8767(94%) |
| people | 1521 | 1 | F3841(94%) |
| hold | 509 | 1 | F9485(94%) |
| late | 509 | 1 | F7674(94%) |
| paw | 510 | 1 | F1845(94%) |
| attractive | 1006 | 1 | F14364(93%) |
| already | 506 | 1 | F12609(93%) |
| asked | 1939 | 1 | F4266(93%) |
| deep | 313 | 1 | F15672(93%) |
| explained | 1008 | 1 | F12479(93%) |
| linda | 2540 | 1 | F9103(92%) |
| thought | 2024 | 1 | F636(92%) |
| happy | 3556 | 1 | F10842(92%) |
| balloons | 505 | 1 | F5942(92%) |
| reality | 506 | 1 | F5493(91%) |
| is | 1558 | 1 | F10310(91%) |
| know | 1527 | 1 | F9104(91%) |
| right | 509 | 1 | F2458(91%) |
| unsure | 507 | 1 | F8975(90%) |
| draw | 509 | 1 | F13247(90%) |
| way | 149 | 1 | F6380(90%) |
| room | 1018 | 1 | F12505(90%) |
| tried | 626 | 1 | F11510(90%) |
| jumped | 507 | 1 | F2898(89%) |
| ears | 512 | 1 | F3113(89%) |
| liked | 508 | 1 | F13180(89%) |
| few | 1530 | 1 | F13086(89%) |
| outside | 2024 | 1 | F8676(89%) |
| parent | 462 | 1 | F12043(88%) |
| face | 509 | 1 | F9582(88%) |
| instead | 1012 | 1 | F5273(88%) |
| words | 508 | 1 | F13386(88%) |
| meant | 1016 | 1 | F6389(88%) |
| daddy | 2540 | 1 | F14788(88%) |
| laughed | 1012 | 1 | F12866(87%) |
| sorry | 1016 | 1 | F10181(87%) |
| full | 1527 | 1 | F8656(86%) |
| touch | 508 | 1 | F10664(86%) |
| attack | 122 | 1 | F8825(86%) |
| told | 1527 | 1 | F7816(85%) |
| who | 1020 | 1 | F1371(85%) |
| nearby | 296 | 1 | F7825(85%) |
| nick | 87 | 1 | F3770(85%) |
| anna | 2036 | 1 | F9896(85%) |
| started | 5566 | 1 | F5332(85%) |
| first | 1524 | 1 | F690(85%) |
| bad | 1511 | 1 | F5676(85%) |
| more | 1242 | 1 | F8388(85%) |
| best | 509 | 1 | F1271(85%) |
| shouted | 1012 | 1 | F5666(84%) |
| lost | 426 | 1 | F8187(84%) |
| wet | 1530 | 1 | F3222(83%) |
| boy | 510 | 1 | F8680(83%) |
| door | 509 | 1 | F6735(83%) |
| hugged | 1014 | 1 | F1400(83%) |
| together | 1515 | 1 | F8256(82%) |
| me | 1059 | 1 | F15996(82%) |
| done | 1018 | 1 | F10400(82%) |
| off | 510 | 1 | F8995(82%) |
| took | 1018 | 1 | F4615(82%) |
| into | 1527 | 1 | F7301(82%) |
| when | 3055 | 1 | F15953(80%) |
| again | 3048 | 1 | F979(80%) |
| made | 1662 | 1 | F7541(80%) |
| heard | 1362 | 1 | F5492(79%) |
| knew | 509 | 1 | F13290(79%) |
| fly | 1020 | 1 | F8913(79%) |
| in | 5211 | 1 | F7899(79%) |
| hat | 24 | 1 | F2557(79%) |
| quickly | 941 | 1 | F13224(79%) |
| decorated | 504 | 1 | F6988(79%) |
| moral | 508 | 1 | F3469(79%) |
| branches | 1010 | 1 | F1125(79%) |
| skill | 508 | 1 | F7745(78%) |
| morning | 506 | 1 | F10152(78%) |
| looked | 1521 | 1 | F15671(78%) |
| ruined | 507 | 1 | F13971(77%) |
| i | 2134 | 1 | F12920(77%) |
| underwater | 228 | 1 | F12945(77%) |
| catch | 508 | 1 | F4089(77%) |
| fast | 245 | 1 | F14104(77%) |
| fishes | 1521 | 1 | F6170(77%) |
| could | 4575 | 1 | F6699(77%) |
| sound | 508 | 1 | F12856(76%) |
| over | 1527 | 1 | F10999(76%) |
| grew | 509 | 1 | F15109(76%) |
| places | 1014 | 1 | F4576(76%) |
| want | 1530 | 1 | F3242(76%) |
| turns | 2032 | 1 | F1117(76%) |
| ittle | 21 | 1 | F9890(76%) |
| wished | 1031 | 1 | F12109(76%) |
| be | 2439 | 1 | F5036(75%) |
| take | 3054 | 1 | F5952(75%) |
| idea | 1018 | 1 | F11197(75%) |
