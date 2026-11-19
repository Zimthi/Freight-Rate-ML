import json
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib import colors
m=json.load(open('metrics.json')); S=getSampleStyleSheet()
def P(t,s='BodyText'): return Paragraph(t,S[s])
def bl(items): return [P('&bull; '+i) for i in items]
doc=SimpleDocTemplate('report.pdf',pagesize=letter,leftMargin=.8*inch,rightMargin=.8*inch,topMargin=.7*inch,bottomMargin=.7*inch)
r=[P('Freight Rate Prediction: Approach &amp; Validation','Title'),
P('Data and issues found','Heading2')]
r+=bl(['Development data: 48,000 loads, 2025-01-01 to 2025-10-31. Final set: 12,000 loads dated 2025-11-01 to 2025-12-31, so the task is forecasting forward in time.',
'292 negative weights: sign errors (|w| matches the positive distribution), fixed with abs(). Weights capped at 47,500 lb (kept).',
'300 missing weights (filled with equipment median + missing flag) and 374 missing market_index values. market_index is a date-level signal plus ~0.025 noise, so missing values are filled with that date\'s mean.',
'About 0.7% of rows (340) have rates at 5-13x the normal $/mile, i.e. corrupt or spike labels. The model uses median (absolute-error) loss on log(rate), so they have little influence and no rows are deleted.',
'8 cities in the final set never appear in training, so city names are not used. Geography enters via lat/lon, distance, circuity (distance / great-circle) and direction.',
'quote_signal is nearly uncorrelated with rate, and removing it improved forward CV MAE (about 141 to 118), so it is excluded. It is also not provided for the December chart.'])
r+=[P('Validation approach','Heading2')]
r+=bl(['Forward-chaining (expanding window) CV: train on everything before month M, test on month M, for Jun-Oct 2025. Random K-fold was rejected because it leaks neighbouring days and overstates accuracy for a future-dated test set.',
'Unseen-city check: 8 random cities removed from training; tested on Sep-Oct loads touching them.'])
rows=[['Test month','MAE ($)','RMSE ($)','MAPE %','MedAPE %']]
for x in m['forward_cv_monthly']: rows.append([x['test_month'],f"{x['MAE']:.1f}",f"{x['RMSE']:.1f}",f"{x['MAPE_%']:.2f}",f"{x['MedAPE_%']:.2f}"])
a=m['forward_cv_mean']; rows.append(['Mean',f"{a['MAE']:.1f}",f"{a['RMSE']:.1f}",f"{a['MAPE_%']:.2f}",f"{a['MedAPE_%']:.2f}"])
u=m['unseen_city_holdout']; rows.append(['Unseen cities',f"{u['MAE']:.1f}",f"{u['RMSE']:.1f}",f"{u['MAPE_%']:.2f}",f"{u['MedAPE_%']:.2f}"])
t=Table(rows,hAlign='LEFT'); t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#064A56')),('TEXTCOLOR',(0,0),(-1,0),colors.white),('GRID',(0,0),(-1,-1),.4,colors.grey),('FONTSIZE',(0,0),(-1,-1),9),('FONTNAME',(0,-3),(-1,-3),'Helvetica-Bold')]))
r+=[Spacer(1,6),t,Spacer(1,6),P('RMSE is dominated by the spike labels (unpredictable), so MAE/MedAPE are the more informative metrics.')]
r+=[P('Model','Heading2')]
r+=bl(['HistGradientBoostingRegressor on log(rate), absolute-error loss, 5-seed average, trained on all 48,000 labeled rows. Features: log distance, equipment, weight, market_index, lat/lon, circuity, direction, day of week, day of month, day of year.',
'Chosen over a linear model/other losses via ablation: median loss beat squared loss; lat/lon and calendar features each reduced error.'])
r+=[P('December chart','Heading2'),
P('Only the date varies for the fixed lane. The model never sees quote_signal, and market_index is set to the daily mean of validation.csv for each December date (feature data only, no labels). Day-of-year is beyond the training range, so the model holds its latest learned level; the small weekly wiggle comes from market_index and day of week.'),
Image('scorer_results/candidate_december.png',width=6.6*inch,height=6.6*inch*__import__('PIL.Image',fromlist=['x']).open('scorer_results/candidate_december.png').size[1]/__import__('PIL.Image',fromlist=['x']).open('scorer_results/candidate_december.png').size[0])]
doc.build(r)
