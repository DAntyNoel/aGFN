数据导引

1. Set Generation
    - DB,FL-DB,TB: Rebuttal-Set-Temp-Old (要设置reward_temp=1)
    - FL-SubTB: 
        - large: Refactored-Alpha-GFN-Set-New-icml （注意设置fl=1，这个wandb project里的SubTB不能用，有问题）
        - small/medium: Rebuttal-Set-FL
    - SubTB: small/medium/large均在 Refactored-Alpha-GFN-Set-New-icml-fl0
    - 需要的指标: modes, mean_top_1000_R, mean_top_1000_similarity, spearman_corr_test
2. Bit Sequence Generation
    - DB,SubTB,TB: 直接复用原来的结果
    - FL-DB,FL-SubTB, k=4/6/8/10: Rebuttal-Bit-FL
    - FL-DB, FL-SubTB, k=2: Refactored-Alpha-GFN-Bitseq-icml2026
    - 需要的指标: modes, spearman_corr_test
3. Set Generation, temperature scaling of reward
    - DB, FL-DB, TB: Rebuttal-Set-Temp-Fixed
    - 需要的指标: modes, mean_top_1000_R, mean_top_1000_similarity, spearman_corr_test
