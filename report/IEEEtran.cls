\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{IEEEtran}[2026/05/31 Minimal local fallback for course draft]
\LoadClass[10pt,twocolumn]{article}
\RequirePackage{times}
\RequirePackage[margin=0.7in]{geometry}
\setlength{\columnsep}{0.24in}
\pagestyle{plain}
\newcommand{\IEEEauthorblockN}[1]{\textbf{#1}\\}
\newcommand{\IEEEauthorblockA}[1]{#1}
\newenvironment{IEEEkeywords}{\par\small\noindent\textbf{Index Terms---}}{\par}
\renewcommand{\maketitle}{%
  \twocolumn[\begin{@twocolumnfalse}\centering\LARGE\bfseries\@title\par\vspace{0.8em}\normalsize\@author\par\vspace{1.0em}\end{@twocolumnfalse}]%
}
