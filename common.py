from tornado import template


def get_max_from_dict(d: dict) -> int:
    ans = max(d)
    if type(ans) == int:
        return ans
    else:
        ans = d[ans]
        if type(ans) == int:
            return ans
        else:
            return 0


def draw_barplot(values: dict, svg_file: str, title: str, x_label='x', y_label='y'):
    loader = template.Loader('.')
    bar_width, x_pos = 18.7, 1
    bar_gap = bar_width / 3
    width, height = (len(values) * bar_width) + ((len(values) - 1) * bar_gap), get_max_from_dict(values) - 75
    translate_y = -height / 10000

    html = loader.load('bar.html').generate(values=values,
                                            bar_width=bar_width,
                                            bar_gap=bar_gap,
                                            x_pos=x_pos,
                                            width=width,
                                            height=height,
                                            translate_y=translate_y,
                                            x_label=x_label,
                                            y_label=y_label,
                                            title=title)
    # f = open('images/bar-template-' + datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S') + '.svg', 'w')

    f = open(svg_file, 'w')
    f.write(html.decode('utf-8'))
    f.close()