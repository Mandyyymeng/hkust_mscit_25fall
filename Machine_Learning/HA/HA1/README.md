## HA1 documents

21194672 ZHANG Zimeng

In HA1, a softmax regression function is implemented. First, we calculate the exp(XW) term for later probability calculations; then, conditional probability matrices are computed and combined to calculate loss, a L2 term is added to the loss. The gradient of weight matrix W is also calculated and added a L2 regression term following the formula provided on page 41 of L03. The meaning of all variables and operations can be found in code comments. 

