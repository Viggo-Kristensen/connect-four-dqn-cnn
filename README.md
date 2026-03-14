This project is my attempt at creating an AI for connect four using Deep Q-learning. 

The Model

The input layer for the model consists of two 7x6 connect four grids which each are a binary map. In the first binary map 1's denote the slots where the agents pieces are where as in the other 1's denote the placement of opponents pieces. These two binary maps are then passed through 3x3 filters through 3 CNN-layers. After this there are a couple linear layers which then outputs 7 Q-values where the ith Q value denotes the value of placing a piece in the ith column.



