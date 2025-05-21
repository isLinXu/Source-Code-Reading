from m4.sourcing.data_collection.processors import DOMTreeSimplificator


html_str = """<!DOCTYPE html>
<html>
    <head>
    </head>
    <body>
        <script>
            Blabla
        </script>
        <weird tag><another weird tag></another weird tag></weird tag>
        <div1>
            <a href="">Hello</a> World!
            Salut!
            <i>Hey!</i>
            <img src="http://img">
            <video><source src="http://video"></video>
            Yoo
            <div>
                <div>
                    <p>Haha</p>
                </div>
                <div>
                    <div>
                        <div>
                            <p>nest</p>
                        </div>
                    </div>
                </div>
            </div>
            <p>Hehe</p>
            <div2>
                <p></p>
            </div2>
            <div class="date">Date</div>
        </div1>
        <div>
            <p>Hoho <!--comment in a tag--></p>
        </div>
    </body><!-- body-wrapper -->
    <!-- comment
    on several lines    and with tab inside

        -->
    <!-- unclosed comment
</html>
"""


def debug():
    tree_simplificator = DOMTreeSimplificator(
        strip_multiple_linebreaks=True,
        strip_multiple_spaces=True,
        remove_html_comments=True,
        replace_line_break_tags=True,
        unwrap_tags=True,
        strip_tags=True,
        strip_special_divs=True,
        remove_dates=True,
        remove_empty_leaves=True,
        unnest_nodes=True,
        remake_tree=True,
    )
    selectolax_tree = tree_simplificator(html_str, type_return="selectolax_tree")
    print(selectolax_tree.html)
    print("-----")
    for node in selectolax_tree.root.traverse():
        print(node.tag)


if __name__ == "__main__":
    debug()
