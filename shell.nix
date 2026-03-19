with import <nixpkgs> {};

mkShell {
  packages = [ gnuplot ];
  LD_LIBRARY_PATH = lib.makeLibraryPath [ libX11 libxkbcommon libXi libGL ];
}
