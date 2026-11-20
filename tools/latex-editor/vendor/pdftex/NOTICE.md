# Browser pdfLaTeX component

`swiftlatexpdftex.js` and `swiftlatexpdftex.wasm` are unmodified distribution files from SwiftLaTeX, copyright SwiftLab and contributors. This release runs pdfTeX 1.40.21. The SwiftLaTeX code and local worker adapter are available under the GNU Affero General Public License v3. The [license text](LICENSE-AGPL.txt) accompanies the program. The authoring tool loads all engine and package bytes from this site. Local worker code replaces upstream network-based package lookup with an in-memory library.

- [Release v20022022](https://github.com/SwiftLaTeX/SwiftLaTeX/releases/tag/v20022022)
- [Corresponding SwiftLaTeX sources and build instructions at 912068d](https://github.com/SwiftLaTeX/SwiftLaTeX/tree/912068d6f746a58c210eec69053ef72e2ed60b4c)
- [Format file, corresponding code and license](https://github.com/SwiftLaTeX/Texlive-Ondemand/tree/cc2146e220658f5ee0f8643355252ae0995fbc98)
- [Local adapter source](worker.js) and [client source](../../engine.js)

`texlive.bin` bundles a compressed library for the preloaded LaTeX2e 2020-02-02 format. [packages.json](packages.json) identifies every installed package, revision, original file path, distribution integrity hash and catalogue license. It selects runtime files from the immutable [TeX Live 2019 final archive](https://ftp.math.utah.edu/pub/tex/historic/systems/texlive/2019/tlnet-final/archive/). For any package `NAME`, this archive supplies `NAME.tar.xz` and applicable `NAME.source.tar.xz` and `NAME.doc.tar.xz` distributions. Files retain their original notices. `pdftex.map` combines the selected Computer Modern and AMS font maps. The content index is followed by the unmodified contents of each file. Run `python3 rebuild.py` to reproduce both engine binaries and the resource pack using the verified inputs.

The licenses listed in the TeX Live catalogue apply per component. Included notices are [TeX Live licensing](licenses/LICENSE.TL), [CTAN licensing](licenses/LICENSE.CTAN), [LaTeX legal information](licenses/legal.txt), [LaTeX Project Public License](licenses/lppl.txt), [LPPL 1.2](licenses/lppl-1-2.txt), and [SIL Open Font License for the AMS Type 1 fonts](licenses/OFL.txt). See the individual file headers and the corresponding TeX Live distributions for the Knuth font, MIT and other component terms.
