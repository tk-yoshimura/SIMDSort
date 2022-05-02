# SIMDSort

AVX sorting experiment

## Requirement
AVX2 suppoted CPU. (Intel:Haswell(2013)-, AMD:Excavator(2015)-)

## Algorithm (YMM 8x32)
![algo](https://github.com/tk-yoshimura/SIMDSort/blob/main/figures/algo.svg)  

## Processing Speed (float)

### Uniform Random Values
x4-10 faster than std::sort  
![random](https://github.com/tk-yoshimura/SIMDSort/blob/main/figures/sort_random_speed_s.svg)  

### Imbalanced Random Values (1%: v &in; [0, 1), 99%: v &in; [0, 0.01))
x4-10 faster than std::sort  
![imbalanced random](https://github.com/tk-yoshimura/SIMDSort/blob/main/figures/sort_inbalance_speed_s.svg)  

### NormalDist Random Values
x4-10 faster than std::sort  
![ndist](https://github.com/tk-yoshimura/SIMDSort/blob/main/figures/sort_ndist_speed_s.svg)  

### Reverse Values
disadvantage to quicksort as the number increase  
![ndist](https://github.com/tk-yoshimura/SIMDSort/blob/main/figures/sort_reverse_speed_s.svg)  

### All Conditions
stable speed regardless of conditions
![avxall](https://github.com/tk-yoshimura/SIMDSort/blob/main/figures/sort_avxall_speed_s.svg)  

## Processing Speed (double)

### Uniform Random Values
x2-4 faster than std::sort  
![random](https://github.com/tk-yoshimura/SIMDSort/blob/main/figures/sort_random_speed_d.svg)  

### Imbalanced Random Values (1%: v &in; [0, 1), 99%: v &in; [0, 0.01))
x2-4 faster than std::sort  
![imbalanced random](https://github.com/tk-yoshimura/SIMDSort/blob/main/figures/sort_inbalance_speed_d.svg)  

### NormalDist Random Values
x2-4 faster than std::sort  
![ndist](https://github.com/tk-yoshimura/SIMDSort/blob/main/figures/sort_ndist_speed_d.svg)  

### Reverse Values
disadvantage to quicksort as the number increase  
![ndist](https://github.com/tk-yoshimura/SIMDSort/blob/main/figures/sort_reverse_speed_d.svg)  

### All Conditions
stable speed regardless of conditions
![avxall](https://github.com/tk-yoshimura/SIMDSort/blob/main/figures/sort_avxall_speed_d.svg)  

## Licence
[CC-BY](https://github.com/tk-yoshimura/SIMDSort/blob/main/LICENSE)

## Author

[T.Yoshimura](https://github.com/tk-yoshimura)
