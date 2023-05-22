const User = require('../models/user');

module.exports = {
  async create(user) {
    return User.create(user);
  },

  async update(id, user) {
    return User.update(user, { where: { id } });
  },

  async findById(id) {
    return User.findByPk(id);
  },

  async findAll() {
    return User.findAll();
  },

  async findByLogin(login) {
    return User.findOne({ where: { login } });
  },

  async delete(id) {
    return User.destroy({ where: { id } });
  },
};
