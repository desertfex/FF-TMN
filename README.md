# FF-TMN
Frequency-driven channel attention-augmented full-scale temporal modeling network for skeleton-based action recognition

# Introduction
The skeleton-based human action recognition has become a popular research focus due to its promising applications. The current methods that model skeletons as spatial–temporal graphs have solidly advanced the state-of-the-art performance. However, there are still two problems to be addressed: (1) Although many existing methods employ multi-scale temporal modeling modules, they are still insufficient to fully capture both short-term and long-term temporal clues. (2) Channel attention modules are often employed for this task to improve recognition accuracy. However, they all use the same strategies to aggregate information from spatial and temporal dimensions, ignoring the fact that the semantic information in these two dimensions is quite different, thus leading to suboptimal network performance. In response to the above problems, we propose a frequency-driven channel attention-augmented full-scale temporal modeling network (FF-TMN) that incorporates two novel modules: (1) An effective and efficient full-scale temporal modeling module (FTMM) encompassing up to three multi-scale modeling strategies is proposed to equip the network with full-scale temporal modeling capabilities. (2) We propose a frequency-driven channel attention module (FCAM) tailored for the HAR task, which is the first module to generate channel descriptors with two different aggregation strategies: global average pooling for the spatial dimension and discrete cosine transform for the temporal dimension. Exhaustive experiments on three challenging large-scale datasets demonstrate that our FF-TMN achieves state-of-the-art performance.
# Citation
Fanjia Li, Aichun Zhu, Juanjuan Li, Yonggang Xu, Yandong Zhang, Hongsheng Yin, Gang Hua,
Frequency-driven channel attention-augmented full-scale temporal modeling network for skeleton-based action recognition,
Knowledge-Based Systems,
Volume 256,
2022,
109854,
ISSN 0950-7051,
https://doi.org/10.1016/j.knosys.2022.109854.
(https://www.sciencedirect.com/science/article/pii/S0950705122009479)
