const express = require('express');
const app = express();
const bodyParser = require('body-parser');
// import routes
const userRoutes = require('./routes/userRoutes');

// middleware
app.use(bodyParser.json());

// routes
app.use('/users', userRoutes);


// handle requests to the base URL
// app.get('/', (req, res) => {
//   res.send('Hello World!');
// });

// start the server
app.listen(3000, () => {
  console.log('online');
});