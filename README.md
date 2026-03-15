This project aims to build an AI agent that plays Connect Four using Deep Q-Learning.
The model’s input consists of two 7×6 Connect Four grids represented as binary maps. In the first map, a value of 1 indicates positions occupied by the agent’s pieces, while 0 represents empty or opponent positions. In the second map, 1 indicates the opponent’s pieces and 0 represents all other positions.

These two binary maps are processed through three convolutional neural network (CNN) layers using 3×3 filters to extract spatial patterns and relationships on the board. The resulting feature representations are then passed through several fully connected (linear) layers.
Finally, the network outputs seven Q-values, corresponding to the seven possible columns in the Connect Four board. The 
i-th Q-value represents the estimated value (expected future reward) of placing a piece in column i
