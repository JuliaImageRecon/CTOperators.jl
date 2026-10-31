# Static export of the interactive figures (WGLMakie through Bonito). Bonito writes the JavaScript and
# the recorded states of a page's figures into `bonito/` in the folder it is given and links them
# relative to the page. Documenter runs the page in the build folder of its Markdown file; with pretty
# URLs (on CI, see make.jl) the HTML of page `name` is `name/index.html`, one folder deeper, so the
# assets have to go there.

using Bonito

pretty_urls() = get(ENV, "CI", "false") == "true"   # as in make.jl

bonito_page(name) = Page(; offline=true, current_page_dir=pretty_urls() ? joinpath(pwd(), name) : pwd())
