from flask import Flask, render_template, request
import football

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
   draft_starters = draft_bench = QB_results = RB_results = WR_results = TE_results = DEF_results = None
   if request.method == 'POST':
      num_teams = int(request.form['num_teams'])
      draft_starters, draft_bench, QB_results, RB_results, WR_results, TE_results, DEF_results = football.main(num_teams)
   else:
      draft_starters, draft_bench, QB_results, RB_results, WR_results, TE_results, DEF_results = football.main(0)
   return render_template('index.html',
                           draft_starters=draft_starters,
                           draft_bench=draft_bench,
                           QB_result=QB_results,
                           RB_result=RB_results,
                           WR_result=WR_results,
                           TE_result=TE_results,
                           DEF_result=DEF_results)

if __name__ == '__main__':
   app.run(debug=True)