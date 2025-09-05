from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats as st

class City:
    def __init__(
        self, population: int = 1000, average_age: int = 20, std_age: int = 50
    ) -> None:
        self.population = population
        self.average_age = average_age
        self.std_age = std_age
        self.id = 0
        self.normal_dist = iter(
            np.random.normal(self.average_age, self.std_age, self.population)
        )

    def get_person_from_city(self):
        self.id += 1
        while True:
            age = next(self.normal_dist)
            if 14 < age < 90:
                break
        sex = np.random.choice(["male", "female"])
        problem_type = np.random.choice(["murder", "theft", "adultery"])
        return Person(self.id, age, sex, problem_type)

    def get_people(self, n):
        return [self.get_person_from_city() for _ in range(n)]

    def __repr__(self) -> str:
        return f"Created city with population of {self.population}"


class Person:
    def __init__(self, id, age, sex, problem_type):
        self.id = id
        self.age = age
        self.sex = sex
        self.problem_type = problem_type


class Detective:
    _id_count = 0

    def __init__(self, config: Dict[str, Dict[str, Any]], name: Optional[str] = None):
        self.id = Detective._id_count
        Detective._id_count += 1
        self.name = name
        self.ncases = {k: 0 for k in config.keys()}
        self.price = {k: item["price"] for k, item in config.items()}
        self.success_rates = {k: item["success_rate"] for k, item in config.items()}
        self.avg_time = {k: item["avg_time"] for k, item in config.items()}


class Case:
    _id_counter = 0

    def __init__(self, person: Person, detective: Detective):
        self.id = Case._id_counter
        Case._id_counter += 1
        self.person = person
        self.detective = detective
        self.problem_type = person.problem_type
        self.price = detective.price[person.problem_type]
        self.detective_id = detective.id

        self.local_id = detective.ncases[person.problem_type]
        self.status = "ongoing"
        self.time_init = self.init_date(datetime(2000, 1, 1), datetime(2025, 12, 31))
        self.time_finish = None
        self.time_spent = None

    def init_date(self, start_date, end_date):
        delta = end_date - start_date
        delta_seconds = delta.total_seconds()
        random_seconds = np.random.uniform(0, delta_seconds)
        return start_date + timedelta(seconds=random_seconds)

    def finish_date(self):
        mean_seconds = timedelta(
            days=self.detective.avg_time[self.problem_type]
        ).total_seconds()
        std_seconds = mean_seconds * 0.1
        finish = np.random.normal(mean_seconds, std_seconds)
        return self.time_init + timedelta(seconds=finish)


class Agency:
    def __init__(self, detectives: list[Detective]):
        self.detectives = detectives
        self.cases = None
        self.case_types = list(detectives[0].ncases.keys()) if detectives else []
        self.case_count = 0
        self.revenue = {detective.id: 0 for detective in detectives}
        self.distribution = {}

    def generate_distributions(self):
        for detective in self.detectives:
            self.distribution[detective.id] = {}
            for case_type in self.case_types:
                self.distribution[detective.id][case_type] = np.random.binomial(
                    n=1,
                    p=detective.success_rates[case_type],
                    size=detective.ncases[case_type],
                )

    def _process_case(self, case):
        case.time_finish = case.finish_date()
        case.time_spent = case.time_finish - case.time_init
        case.status = self.distribution[case.detective.id][case.problem_type][
            case.local_id
        ]
        case.revenue = (
            None if isinstance(case.status, str) else case.status * case.price
        )
        self.revenue[case.detective.id] += case.detective.price[case.problem_type]

    def process_cases(self):
        if not self.cases:
            return
        try:
            for case in self.cases:
                self._process_case(case)
        except Exception as err:
            print(err)

    def get_cases(self, people):
        self.clients=people
        self.cases = [self._assign_case(person) for person in people]
        self.generate_distributions()

    def _assign_case(self, person):
        detective_id = np.random.choice([0, 1])
        case = Case(person, self.detectives[detective_id])
        self.detectives[detective_id].ncases[person.problem_type] += 1
        self.case_count += 1
        return case

    def get_dataframe(self):
        self.df = pd.DataFrame(
            {
                # 'case id': [case.id for case in cases],
                "status": [case.status for case in self.cases],
                "init_date": [case.time_init for case in self.cases],
                "finish_date": [case.time_finish for case in self.cases],
                "time_spent": [case.time_spent for case in self.cases],
                "detective_id": [case.detective_id for case in self.cases],
                "revenue": [case.revenue for case in self.cases],
                "case_type": [case.problem_type for case in self.cases],
            },
            index=[case.id for case in self.cases],
        )
        return self.df


class QuickStart:
    def __init__(
        self,
        rm=dict(
            murder=dict(price=2000, avg_time=2, success_rate=0.5),
            theft=dict(price=500, avg_time=3, success_rate=0.5),
            adultery=dict(price=800, avg_time=1, success_rate=0.5),
        ),
        sh=dict(
            murder=dict(price=2000, avg_time=10, success_rate=0.5),
            theft=dict(price=500, avg_time=10, success_rate=0.5),
            adultery=dict(price=800, avg_time=10, success_rate=0.5),
        ),
        population=10000,
        avg_age=30,
    ) -> None:
        self.config_RM = rm

        self.config_SH = sh

        self.town = City(population=population, average_age=avg_age)

        self.clients = self.town.get_people(1000)
        self.agency = Agency(
            [
                Detective(config=self.config_SH, name="SH"),
                Detective(config=self.config_RM, name="RM"),
            ]
        )

    def go(self):
        self.agency.get_cases(self.clients)
        self.agency.process_cases()
        return self.agency




class Statistics:
    def __init__(self, results) -> None:
        self.results = results
        self.names = list(results.keys())
        self.conditions = [v['condition'] for k, v in results.items()]
        self.effect = [v['effect'] for k, v in results.items()]
        self.hist = [v['hist'] for k, v in results.items()]

    def show(self, ttest=True):
        n = len(self.names)
        if not ttest:
            # plt.title('H0: Detective 1 is better then Detective 2')
            fig, axs = plt.subplots(3, n, figsize=(3 * n, 5))# gridspec_kw={'height_ratios': [1, 1],'width_ratios': [1, 1] })
            fig.suptitle('H0: Detective 1 is better than Detective 2')
            
            # Plot histograms on the top row, one per box
            for i, hist_data in enumerate(self.hist):
                sns.histplot(hist_data, ax=axs[0, i], bins=10, color='grey')
                axs[0,i].axvline(hist_data.quantile(0.025),color='grey',ls='--')
                axs[0,i].axvline(hist_data.quantile(0.975),color='grey',ls='--')
                axs[0,i].axvline(self.effect[i],color='red',ls='--')
                axs[0, i].set_title(self.names[i])
                # axs[0, i].set_xticks([])
                axs[0, i].set_yticks([])
            
            for i, hist_data in enumerate(self.hist):
                st.probplot(hist_data, dist='norm', plot=axs[2,i])
            
        # Bottom row: colored boxes
            colors = ['green' if c else 'red' for c in self.conditions]
            x = np.arange(n)
            y = np.zeros(n)
            for j in range(n):
                sns.scatterplot(x=[x[j]], y=[y[j]], s=10000, color=colors[j], marker='s', edgecolor='none', ax=axs[1, j])
                sns.despine(ax=axs[1, j], left=True, bottom=True, right=True, top=True)
                axs[1, j].set_xlim(-0.1, n - 0.1)
                axs[1, j].set_ylim(-0.1, 0.1)
                axs[1, j].set_yticks([])
                axs[1, j].set_xticks([])
            

            # Hide all other axes on bottom row except the first and set limits
            for i in range(1, n):
                axs[1, i].axis('off')
        else:
            fig, axs = plt.subplots(1, n, figsize=(1 * n, 2))# gridspec_kw={'height_ratios': [1, 1],'width_ratios': [1, 1] })
            fig.suptitle('H0: Detective 1 is better than Detective 2')

            # Bottom row: colored boxes
            colors = ['green' if c else 'red' for c in self.conditions]
            x = np.arange(n)
            y = np.zeros(n)
            for j in range(n):
                sns.scatterplot(x=[x[j]], y=[y[j]], s=10000, color=colors[j], marker='s', edgecolor='none', ax=axs[ j])
                sns.despine(ax=axs[j], left=True, bottom=True, right=True, top=True)
                axs[ j].set_xlim(-0.1, n - 0.1)
                axs[ j].set_ylim(-0.1, 0.1)
                axs[ j].set_yticks([])
                axs[ j].set_xticks([])
            

            # Hide all other axes on bottom row except the first and set limits
            for i in range(1, n):
                axs[ i].axis('off')



        plt.tight_layout()
        plt.show()

        

class Metrics:
    def __init__(self,df) -> None:
        self.df=df
        pass
        
    def single_metrics_bootstrap(self, name='revenue', agg='mean', two_sided=True):
        rm = self.df[self.df['detective_id']==0]
        sh = self.df[self.df['detective_id']==1]
        rm_means=pd.Series([rm.sample(frac=1, replace=True)[name].agg(agg) for _ in range(100)])
        sh_means=pd.Series([sh.sample(frac=1, replace=True)[name].agg(agg) for _ in range(100)])
        
        effect=sh_means.mean()-rm_means.mean()
        # print(effect)
        diffs= sh_means-rm_means-effect
        if two_sided:
            p_value = np.mean(np.abs(diffs) >= np.abs(effect))
            condition=True if p_value < 0.05 else False
            return condition, effect, diffs
        else:
            p_value = np.mean(diffs >= effect)
            condition=True if p_value > 0.95 else False
            return condition,effect, diffs
    def single_metrics_ttest(self, name='revenue', agg='mean', two_sided=True):
        rm = self.df[self.df['detective_id']==0]
        sh = self.df[self.df['detective_id']==1]
        effect, p_value = st.ttest_ind(sh[name], rm[name])
        condition=True if p_value < 0.05 else False
        return  condition,effect
    
    def get_bootstrap(self, names=[]):
        self.names=names
        # print(names)
        self.results=dict()
        for name, agg in names:
            # print(name)
            condition, effect,hist=self.single_metrics_bootstrap(name=name, agg=agg)
            self.results[name]={'condition':condition,"effect":effect, 'hist':hist} 
        return Statistics(self.results)

    def get_ttest(self, names=[]):
        self.names=names
        # print(names)
        self.results=dict()
        for name, agg in names:
            # print(name)
            condition, effect=self.single_metrics_ttest(name=name, agg=agg)
            self.results[name]={'condition':condition,"effect":effect, 'hist':None} 
        return Statistics(self.results)
