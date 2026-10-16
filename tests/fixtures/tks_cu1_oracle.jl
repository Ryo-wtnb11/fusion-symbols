# Prints TensorKitSectors' CU1Irrep data on the labels 0+, 0-, j = 1/2..5/2,
# keyed `twice(j):s`: dual, dim, FS phase/indicator, twist per label; fusion
# channels and every nonzero Nsymbol, and every Rsymbol / Fsymbol whose bits
# are not +0.0 (so signed zeros are recorded), over all tuples, as IEEE-754 bits. Run with TensorKitSectors 0.3.9 in the active
# project: `julia tks_cu1_oracle.jl > tks-0.3.9-cu1-values.txt`.
using TensorKitSectors; const T = TensorKitSectors
using TensorKitSectors: CU1Irrep, HalfInt, twice
bits(x) = string(reinterpret(UInt64, Float64(x)), base = 16, pad = 16)
lab(a) = string(twice(a.j), ":", a.s)
vs = [CU1Irrep(0, 0), CU1Irrep(0, 1), (CU1Irrep(HalfInt(k // 2)) for k in 1:5)...]
println("# TensorKitSectors ", pkgversion(T), " Julia ", VERSION, "; ",
        T.FusionStyle(CU1Irrep), " ", T.BraidingStyle(CU1Irrep))
for a in vs
    println("dual ", lab(a), " ", lab(dual(a)))
    println("dim ", lab(a), " ", bits(dim(a)))
    println("fs ", lab(a), " ", bits(frobenius_schur_phase(a)), " ", frobenius_schur_indicator(a))
    println("twist ", lab(a), " ", bits(twist(a)))
end
for a in vs, b in vs
    println("fusion ", lab(a), " ", lab(b), " ", join(map(lab, collect(a ⊗ b)), ","))
end
for a in vs, b in vs, c in vs
    n = Nsymbol(a, b, c)
    n != 0 && println("N ", lab(a), " ", lab(b), " ", lab(c), " ", Int(n))
    r = Rsymbol(a, b, c)
    (!iszero(r) || signbit(r)) && println("R ", lab(a), " ", lab(b), " ", lab(c), " ", bits(r))
end
for a in vs, b in vs, c in vs, d in vs, e in vs, f in vs
    x = Fsymbol(a, b, c, d, e, f)
    (!iszero(x) || signbit(x)) && println("F ", join(map(lab, (a, b, c, d, e, f)), " "), " ", bits(x))
end
