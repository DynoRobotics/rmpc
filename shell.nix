with import <nixpkgs> {};

mkShell {
  LD_LIBRARY_PATH = lib.makeLibraryPath (with xorg; [ libX11 libxkbcommon libXi libGL ]);
}
