## How to use our code



Sam optimizer defined in sam.py
How to use SAM optimizer is in main_prune_train.py(line 544)
```
elif args.optmzr == 'sgd-sam':
    base_optimizer = torch.optim.SGD
    optimizer = SAM(model.parameters(), base_optimizer, rho=args.sam_rho, adaptive=args.adaptive, lr=args.lr,v2=args.sam_v2, momentum=args.momentum, weight_decay=args.weight_decay)
```
### 2. Run the code

