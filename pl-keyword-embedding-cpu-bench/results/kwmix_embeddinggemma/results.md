| model | space | dim | r | #clusters | homog. | compl. | AMI | ARI | kw/s (CPU) | ms/kw | load s |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| tfidf (baseline) | lexical | - | - | 83 | 0.983 | 0.636 | 0.471 | 0.328 | - | - | - |
| embeddinggemma | raw+L2 | 768 | 0.5 | 7 | 0.598 | 0.894 | 0.668 | 0.387 | 101.3 | 9.88 | 1.0 |
| embeddinggemma | raw+L2 | 768 | 1.0 | 11 | 0.812 | 0.949 | 0.840 | 0.663 | 101.3 | 9.88 | 1.0 |
| embeddinggemma | raw+L2 | 768 | 2.0 | 14 | 0.923 | 0.950 | 0.914 | 0.861 | 101.3 | 9.88 | 1.0 |
| embeddinggemma | raw+L2 | 768 | 4.0 | 18 | 0.963 | 0.931 | 0.924 | 0.902 | 101.3 | 9.88 | 1.0 |
| embeddinggemma | bg:embgemma_pl_kwmix900k_mrl768 | 768 | 0.5 | 9 | 0.724 | 0.944 | 0.780 | 0.534 | 101.3 | 9.88 | 1.0 |
| embeddinggemma | bg:embgemma_pl_kwmix900k_mrl768 | 768 | 1.0 | 12 | 0.821 | 0.909 | 0.821 | 0.703 | 101.3 | 9.88 | 1.0 |
| embeddinggemma | bg:embgemma_pl_kwmix900k_mrl768 | 768 | 2.0 | 15 | 0.907 | 0.913 | 0.876 | 0.829 | 101.3 | 9.88 | 1.0 |
| embeddinggemma | bg:embgemma_pl_kwmix900k_mrl768 | 768 | 4.0 | 16 | 0.934 | 0.928 | 0.904 | 0.868 | 101.3 | 9.88 | 1.0 |
