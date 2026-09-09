from pathlib import Path

from reportlab.pdfgen import canvas


OUTPUT = Path(__file__).with_name("mi_saturation_panels.pdf")
PAGE_W = 1000
PAGE_H = 370

BLUE = (0.071, 0.247, 0.573)
TEXT = (0.125, 0.153, 0.200)
GRID = (0.86, 0.88, 0.91)


def draw_panel(
    pdf,
    x0,
    y0,
    width,
    height,
    title,
    xs,
    ys,
    x_ticks,
    y_ticks,
    y_min,
    y_max,
    x_label,
):
    plot_left = x0 + 72
    plot_right = x0 + width - 18
    plot_bottom = y0 + 62
    plot_top = y0 + height - 50

    pdf.setFillColorRGB(*TEXT)
    pdf.setFont("Helvetica-Bold", 30)
    pdf.drawCentredString(x0 + width / 2, y0 + height - 30, title)

    x_min = min(x_ticks)
    x_max = max(x_ticks)

    def px(value):
        return plot_left + (value - x_min) * (plot_right - plot_left) / (x_max - x_min)

    def py(value):
        return plot_bottom + (value - y_min) * (plot_top - plot_bottom) / (y_max - y_min)

    pdf.setLineWidth(1.2)
    pdf.setStrokeColorRGB(*GRID)
    for value in y_ticks:
        y = py(value)
        pdf.line(plot_left, y, plot_right, y)
    for value in x_ticks:
        x = px(value)
        pdf.line(x, plot_bottom, x, plot_top)

    pdf.setStrokeColorRGB(*TEXT)
    pdf.setLineWidth(2)
    pdf.line(plot_left, plot_bottom, plot_right, plot_bottom)
    pdf.line(plot_left, plot_bottom, plot_left, plot_top)

    pdf.setFont("Helvetica", 25)
    for value in x_ticks:
        x = px(value)
        pdf.drawCentredString(x, plot_bottom - 30, str(value))
    for value in y_ticks:
        y = py(value)
        pdf.drawRightString(plot_left - 12, y - 8, f"{value:.1f}")

    pdf.setStrokeColorRGB(*BLUE)
    pdf.setFillColorRGB(*BLUE)
    pdf.setLineWidth(4)
    points = [(px(x), py(y)) for x, y in zip(xs, ys)]
    path = pdf.beginPath()
    path.moveTo(*points[0])
    for point in points[1:]:
        path.lineTo(*point)
    pdf.drawPath(path)
    for x, y in points:
        pdf.circle(x, y, 6, stroke=0, fill=1)

    pdf.setFillColorRGB(*TEXT)
    pdf.setFont("Helvetica", 26)
    pdf.drawCentredString(x0 + width / 2, y0 + 10, x_label)


def main():
    pdf = canvas.Canvas(str(OUTPUT), pagesize=(PAGE_W, PAGE_H), pageCompression=1)
    pdf.setTitle("Binary MI saturation and concept aggregation")

    pdf.saveState()
    pdf.setFillColorRGB(*TEXT)
    pdf.setFont("Helvetica", 26)
    pdf.translate(20, PAGE_H / 2)
    pdf.rotate(90)
    pdf.drawCentredString(0, 0, "Channels at maximum MI (%)")
    pdf.restoreState()

    draw_panel(
        pdf,
        18,
        0,
        474,
        PAGE_H,
        "(a) Sample count",
        [5, 10, 20, 50],
        [98.5094, 98.0369, 97.6294, 97.1658],
        [5, 10, 20, 50],
        [97.0, 97.5, 98.0, 98.5],
        96.9,
        98.65,
        "Pseudo-samples per class, n",
    )
    draw_panel(
        pdf,
        510,
        0,
        474,
        PAGE_H,
        "(b) Concept aggregation",
        [1, 2, 3, 4, 10],
        [98.5094, 99.9739, 99.9997, 100.0, 100.0],
        [1, 2, 3, 4, 10],
        [98.5, 99.0, 99.5, 100.0],
        98.4,
        100.05,
        "Target concepts in maximum, m",
    )

    pdf.showPage()
    pdf.save()
    print(OUTPUT)


if __name__ == "__main__":
    main()
