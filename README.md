# SIMDSort

AVX sorting experiment

## Requirement
AVX2 suppoted CPU. (Intel:Haswell(2013)-, AMD:Excavator(2015)-)

## Algorithm (YMM 8x32)
![algo](https://github.com/tk-yoshimura/SIMDSort/blob/main/figures/algo.svg)  

## Processing Speed (float)

### Uniform Random Values
![random](https://github.com/tk-yoshimura/SIMDSort/blob/main/figures/sort_short_random_speed_s.svg)  
![random](https://github.com/tk-yoshimura/SIMDSort/blob/main/figures/sort_random_speed_s.svg)  

### Imbalanced Random Values (1%: v &in; [0, 1), 99%: v &in; [0, 0.01))
![imbalanced random](https://github.com/tk-yoshimura/SIMDSort/blob/main/figures/sort_short_inbalance_speed_s.svg)  
![imbalanced random](https://github.com/tk-yoshimura/SIMDSort/blob/main/figures/sort_inbalance_speed_s.svg)  

### NormalDist Random Values
![ndist](https://github.com/tk-yoshimura/SIMDSort/blob/main/figures/sort_short_ndist_speed_s.svg)  
![ndist](https://github.com/tk-yoshimura/SIMDSort/blob/main/figures/sort_ndist_speed_s.svg)  

### Reverse Values
![reverse](https://github.com/tk-yoshimura/SIMDSort/blob/main/figures/sort_short_reverse_speed_s.svg)  
![reverse](https://github.com/tk-yoshimura/SIMDSort/blob/main/figures/sort_reverse_speed_s.svg)  

### All Conditions
![avxall](https://github.com/tk-yoshimura/SIMDSort/blob/main/figures/sort_short_avxall_speed_s.svg)  
![avxall](https://github.com/tk-yoshimura/SIMDSort/blob/main/figures/sort_avxall_speed_s.svg)  

## Processing Speed (double)

### Uniform Random Values
![random](https://github.com/tk-yoshimura/SIMDSort/blob/main/figures/sort_short_random_speed_d.svg)  
![random](https://github.com/tk-yoshimura/SIMDSort/blob/main/figures/sort_random_speed_d.svg)  

### Imbalanced Random Values (1%: v &in; [0, 1), 99%: v &in; [0, 0.01))
![imbalanced random](https://github.com/tk-yoshimura/SIMDSort/blob/main/figures/sort_short_inbalance_speed_d.svg)  
![imbalanced random](https://github.com/tk-yoshimura/SIMDSort/blob/main/figures/sort_inbalance_speed_d.svg)  

### NormalDist Random Values
![ndist](https://github.com/tk-yoshimura/SIMDSort/blob/main/figures/sort_short_ndist_speed_d.svg)  
![ndist](https://github.com/tk-yoshimura/SIMDSort/blob/main/figures/sort_ndist_speed_d.svg)  

### Reverse Values
![reverse](https://github.com/tk-yoshimura/SIMDSort/blob/main/figures/sort_short_reverse_speed_d.svg)  
![reverse](https://github.com/tk-yoshimura/SIMDSort/blob/main/figures/sort_reverse_speed_d.svg)  

### All Conditions
![avxall](https://github.com/tk-yoshimura/SIMDSort/blob/main/figures/sort_short_avxall_speed_d.svg)  
![avxall](https://github.com/tk-yoshimura/SIMDSort/blob/main/figures/sort_avxall_speed_d.svg)  

## Licence
[CC-BY](https://github.com/tk-yoshimura/SIMDSort/blob/main/LICENSE)

## Author

[T.Yoshimura](https://github.com/tk-yoshimura)
