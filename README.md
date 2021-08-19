# music-generation-tools
Toolbox for generating music

# usage
See example folder

# samples

Routing transformer trained on pop909: https://soundcloud.com/user-419192262-663004693/sets/routing-transformer-pop909
Transformer trained on Lakh midi dataset: https://soundcloud.com/user-419192262-663004693/sets/generated-by-music-transformer-from-scratch

# thanks to 

Great transformers implementations from lucidrains
* https://github.com/lucidrains/reformer-pytorch
* https://github.com/lucidrains/x-transformers
* https://github.com/lucidrains/routing-transformer

Pop music transformer and REMI format
* https://github.com/YatingMusic/remi

Compound word transformer
* https://github.com/YatingMusic/compound-word-transformer

Pop909 dataset
* https://github.com/music-x-lab/POP909-Dataset

Lakh midi dataset
* https://colinraffel.com/projects/lmd/

# Issues

There are still some issues with the reformer model implementation.
It often does not learn how to generate the beginning of the songs well.
This is still a work in progress.
