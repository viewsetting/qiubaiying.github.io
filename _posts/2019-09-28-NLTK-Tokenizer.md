---
layout:     post
title:      NLTK Tokenizer
subtitle:   notes
date:       2019-09-28
author:     viewsetting
header-img: img/NLTK.png
catalog: true
tags:
	- note
	- NLTK
---

# nltk.tokenize

To be updated...

## sent_tokenize

***nltk.tokenize.sent_tokenize(context)***

code:

Tokenize a paragraph in sentence level, then shuffle it.

```python
def shuffle_context(context):
    sent_tokenize_list = nltk.tokenize.sent_tokenize(context)
    print(len(sent_tokenize_list))
    random.shuffle(sent_tokenize_list )
    print(sent_tokenize_list)
    return sent_tokenize_list
```

Tennessee (i/tɛnᵻˈsiː/) (Cherokee: ᏔᎾᏏ, Tanasi) is a state located in the southeastern United States. Tennessee is the 36th largest and the 17th most populous of the 50 United States. Tennessee is bordered by Kentucky and Virginia to the north, North Carolina to the east, Georgia, Alabama, and Mississippi to the south, and Arkansas and Missouri to the west. The Appalachian Mountains dominate the eastern part of the state, and the Mississippi River forms the state's western border. Tennessee's capital and second largest city is Nashville, which has a population of 601,222. Memphis is the state's largest city, with a population of 653,450.

['Tennessee is the 36th largest and the 17th most populous of the 50 United States.', "Tennessee's capital and second largest city is Nashville, which has a population 
of 601,222.", 'Tennessee (i/tɛnᵻˈsiː/) (Cherokee: ᏔᎾᏏ, Tanasi) is a state located in the southeastern United States.', 'Tennessee is bordered by Kentucky and Virginia to the north, North Carolina to the east, Georgia, Alabama, and Mississippi to the south, and Arkansas and Missouri to the west.', "Memphis is the state's largest city, 
with a population of 653,450.", "The Appalachian Mountains dominate the eastern part of the state, and the Mississippi River forms the state's western border."]   
