# Prints TensorKitSectors' FibonacciAnyon and IsingAnyon data as IEEE-754
# bits, labels one-based in `values(I)` order (Fibonacci: 1 = I, 2 = τ;
# Ising: 1 = I, 2 = σ, 3 = ψ): dual, dim, FS phase, twist per label; every
# nonzero Nsymbol, Rsymbol (re im) and Fsymbol over all tuples. Run with
# TensorKitSectors 0.3.9 in the active project:
# `julia tks_anyons_oracle.jl > tks-0.3.9-anyons-values.txt`.
using TensorKitSectors; const T = TensorKitSectors
bits(x) = string(reinterpret(UInt64, Float64(x)), base = 16, pad = 16)
cbits(z) = string(bits(real(z)), " ", bits(imag(z)))
println("# TensorKitSectors ", pkgversion(T), " Julia ", VERSION)
for (name, I) in (("fib", FibonacciAnyon), ("ising", IsingAnyon))
    vs = collect(values(I))
    id(a) = findfirst(==(a), vs)
    for a in vs
        println(name, " dual ", id(a), " ", id(dual(a)))
        println(name, " dim ", id(a), " ", bits(dim(a)))
        println(name, " fs ", id(a), " ", bits(frobenius_schur_phase(a)))
        println(name, " twist ", id(a), " ", cbits(twist(a)))
    end
    for a in vs, b in vs, c in vs
        Nsymbol(a, b, c) != 0 && println(name, " N ", id(a), " ", id(b), " ", id(c), " ", Int(Nsymbol(a, b, c)))
        r = Rsymbol(a, b, c)
        iszero(r) || println(name, " R ", id(a), " ", id(b), " ", id(c), " ", cbits(r))
    end
    for a in vs, b in vs, c in vs, d in vs, e in vs, f in vs
        x = Fsymbol(a, b, c, d, e, f)
        iszero(x) || println(name, " F ", join(map(id, (a, b, c, d, e, f)), " "), " ", bits(x))
    end
end
