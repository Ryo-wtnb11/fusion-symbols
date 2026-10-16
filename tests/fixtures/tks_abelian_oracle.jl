using TensorKit; using TensorKit.TensorKitSectors; const TensorKitSectors = TensorKit.TensorKitSectors
const T = TensorKitSectors
for (nm, I) in (("Z2", Z2Irrep), ("Z3", ZNIrrep{3}), ("Z4", ZNIrrep{4}), ("fZ2", FermionParity))
  vs = collect(values(I))
  for a in vs
    println(nm, " a=", a, " dual=", dual(a), " dim=", repr(dim(a)), " fs=", repr(frobenius_schur_phase(a)), " fsi=", frobenius_schur_indicator(a), " twist=", repr(twist(a)), " ", T.BraidingStyle(I), " ", T.fusionscalartype(I), " ", T.braidingscalartype(I))
  end
  for a in vs, b in vs, c in vs
    r = Rsymbol(a,b,c); r != Nsymbol(a,b,c) && println(nm, " R ", a, b, c, "=", r, " N=", Nsymbol(a,b,c))
  end
  nz = 0; for a in vs, b in vs, c in vs, d in vs, e in vs, f in vs
    F = Fsymbol(a,b,c,d,e,f); N = Nsymbol(a,b,e)*Nsymbol(e,c,d)*Nsymbol(b,c,f)*Nsymbol(a,f,d)
    F != N && println(nm, " F mismatch"); nz += F
  end
  println(nm, " F nonzero count=", nz)
end
u = U1Irrep
for q in (-2,-1,0,1,2)
  a=u(q); println("U1 ", q, " dual=", dual(a), " dim=", repr(dim(a)), " fs=", repr(frobenius_schur_phase(a)), " twist=", repr(twist(a)), " fuse+1=", first(a ⊗ u(1)))
end
println("U1 F ", Fsymbol(u(1),u(1),u(-1),u(1),u(2),u(0)), " ", Fsymbol(u(1),u(1),u(-1),u(1),u(2),u(1)), " R ", Rsymbol(u(1),u(2),u(3)), " ", Rsymbol(u(1),u(2),u(2)), " ", T.BraidingStyle(u), " ", T.fusionscalartype(u))
println(pkgversion(TensorKitSectors), " ", pathof(TensorKitSectors), " ", VERSION)
