{
  description = "MusicEmotionRecon shell";

  inputs = {
    nixpkgs.url = "github:NixOs/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = {
    self,
    nixpkgs,
    flake-utils,
  }:
    flake-utils.lib.eachDefaultSystem (
      system: let
        pkgs = import nixpkgs {inherit system;};
      in {
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            (python314.withPackages (ps:
              with ps; [
                numpy
                matplotlib
                pandas
                seaborn
                parquet
                scipy
                pyarrow
                scikit-image
                jupytext
                ipykernel
              ]))
          ];
        };
      }
    );
}
