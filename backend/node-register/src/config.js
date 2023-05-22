require('dotenv').config();

module.exports = {
  secret: process.env.JWT_SECRET,
  db: {
    host: process.env.DB_HOST,
    port: process.env.DB_PORT,
    database: process.env.DB_DATABASE,
    username: process.env.DB_USERNAME,
    password: process.env.DB_PASSWORD,
    dialect: 'postgres',
    dialectOptions: {
      ssl: process.env.NODE_ENV === 'production',
    },
  },
};
