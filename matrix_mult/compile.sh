# emu
#aoc -v --board s5_ref device/matrix_mult.cl -o bin/matrix_mult_emu.aocx --report
# capi
aoc -v --big-endian --board 385_a7_capi device/matrix_mult.cl -o bin/matrix_mult.aocx --report
