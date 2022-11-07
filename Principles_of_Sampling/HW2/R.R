var_p <- (1 - 1000 / 10000) * (0.035 * (1 - 0.035)) / 999
sqrt(var_p)

str_n <- c(2000, 6000, 2000)
sample_n <- c(100, 500, 400)
p <- c(0.25, 0.014, 0.0075)

var_t <- (1 - sample_n / str_n) * p * (1 - p) / (sample_n - 1)
var_t
sum(var_t)
sqrt(sum(var_t))

sum(c(25, 1.4, 0.75) * c(0.2, 0.6, 0.2))

sum(c(2000, 10000) * c(32.4, 46.8)) / 12000

6 / 40 * 1200

exp <-
  c(
    3,
    3,
    2,
    2,
    3,
    1,
    1,
    1,
    3,
    3,
    1,
    3,
    2,
    3,
    3,
    3,
    3,
    3,
    3,
    2,
    2,
    1,
    3,
    2,
    3,
    3,
    3,
    3,
    3,
    1,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3
  )
fvc <-
  c(
    81,
    64,
    85,
    91,
    60,
    97,
    82,
    99,
    96,
    91,
    71,
    88,
    84,
    85,
    77,
    76,
    62,
    67,
    91,
    99,
    70,
    64,
    72,
    72,
    95,
    96,
    62,
    67,
    95,
    87,
    84,
    89,
    89,
    65,
    67,
    69,
    80,
    98,
    65,
    84
  )

28/40

0.15*mean(fvc[exp==1])+0.15*mean(fvc[exp==2])+0.7*mean(fvc[exp==3])

var_ppp <- c(var(fvc[exp==1]),var(fvc[exp==2]),var(fvc[exp==3]))

ppp <- c(0.15,0.15,0.7)
nnn <- c(6,6,28)
sqrt(sum(ppp^2*var_ppp/nnn*(174/180)))

10000/130


cccc <- c(22.5,17.5,15)
nn <- c(10000,20000,5000)
sum(nn/sqrt(cccc))

propor <- c(10000/sqrt(22.5)/sum(nn/sqrt(cccc)), 20000/sqrt(17.5)/sum(nn/sqrt(cccc)), 5000/sqrt(15)/sum(nn/sqrt(cccc)))

10000/sum(propor*cccc)

543.654*5000/sqrt(15)/sum(nn/sqrt(cccc))

(1-40/1200)*var(fvc)/40
sqrt((1-40/1200)*var(fvc)/40)

1/12*(mean(fvc[exp==1])+mean(fvc[exp==2]))+10/12*mean(fvc[exp==3])

sum((c(100,100,1000)/1200)*(var_ppp/nnn))


1/40*(1-40/1200)*sum(var_ppp*c(100,100,1000)/1200)+1/1600*sum(var_ppp*(1-c(6,6,28)/1200))
